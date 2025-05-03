instance ContinuousMultilinearMap.instContinuousEval :
    ContinuousEval (ContinuousMultilinearMap 𝕜 E F) (Π i, E i) F where
  continuous_eval := by
    /-
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : Finite ι
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : TopologicalSpace F
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalAddGroup F
      inst✝ : Module 𝕜 F
      ⊢ Continuous fun fx => fx.1 fx.2
    -/
    cases nonempty_fintype ι
    /-
      case intro
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : Finite ι
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : TopologicalSpace F
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalAddGroup F
      inst✝ : Module 𝕜 F
      val✝ : Fintype ι
      ⊢ Continuous fun fx => fx.1 fx.2
    -/
    let _ := TopologicalAddGroup.toUniformSpace F
    /-
      case intro
      𝕜 : Type u_1
      ι : Type u_2
      E : ι → Type u_3
      F : Type u_4
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : Finite ι
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : TopologicalSpace F
      inst✝² : AddCommGroup F
      inst✝¹ : TopologicalAddGroup F
      inst✝ : Module 𝕜 F
      val✝ : Fintype ι
      x✝ : UniformSpace F := TopologicalAddGroup.toUniformSpace F
      ⊢ Continuous fun fx => fx.1 fx.2
    -/
    have := comm_topologicalAddGroup_is_uniform (G := F)
    refine (UniformOnFun.continuousOn_eval₂ fun m ↦ ?_).comp_continuous
      (isEmbedding_toUniformOnFun.continuous.prodMap continuous_id) fun (f, x) ↦ f.cont.continuousAt
    exact ⟨ball m 1, NormedSpace.isVonNBounded_of_isBounded _ isBounded_ball,
      ball_mem_nhds _ one_pos⟩


@[deprecated (since := "2024-10-05")]
protected alias ContinuousMultilinearMap.continuous_eval := continuous_eval


lemma continuous_uncurry_of_multilinear (f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F) :
    Continuous (fun (p : G × (Π i, E i)) ↦ f p.1 p.2) := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    F : Type u_4
    inst✝¹¹ : NormedField 𝕜
    inst✝¹⁰ : Finite ι
    inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁷ : TopologicalSpace F
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : TopologicalAddGroup F
    inst✝⁴ : Module 𝕜 F
    G : Type u_5
    inst✝³ : AddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : Module 𝕜 G
    inst✝ : ContinuousConstSMul 𝕜 F
    f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E F)
    ⊢ Continuous fun p => (f p.1) p.2
  -/
  fun_prop
  /-
    🎉 no goals
  -/


lemma continuousOn_uncurry_of_multilinear (f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F) {s} :
    ContinuousOn (fun (p : G × (Π i, E i)) ↦ f p.1 p.2) s :=
  f.continuous_uncurry_of_multilinear.continuousOn


lemma continuousAt_uncurry_of_multilinear (f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F) {x} :
    ContinuousAt (fun (p : G × (Π i, E i)) ↦ f p.1 p.2) x :=
  f.continuous_uncurry_of_multilinear.continuousAt


lemma continuousWithinAt_uncurry_of_multilinear (f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E F) {s x} :
    ContinuousWithinAt (fun (p : G × (Π i, E i)) ↦ f p.1 p.2) s x :=
  f.continuous_uncurry_of_multilinear.continuousWithinAt


/-- If `f` is a continuous multilinear map on `E`
and `m` is an element of `∀ i, E i` such that one of the `m i` has norm `0`,
then `f m` has norm `0`.

Note that we cannot drop the continuity assumption because `f (m : Unit → E) = f (m ())`,
where the domain has zero norm and the codomain has a nonzero norm
does not satisfy this condition. -/
lemma norm_map_coord_zero (f : MultilinearMap 𝕜 E G) (hf : Continuous f)
    {m : ∀ i, E i} {i : ι} (hi : ‖m i‖ = 0) : ‖f m‖ = 0 := by
  classical
  rw [← inseparable_zero_iff_norm] at hi ⊢
  have : Inseparable (update m i 0) m := inseparable_pi.2 <|
    (forall_update_iff m fun i a ↦ Inseparable a (m i)).2 ⟨hi.symm, fun _ _ ↦ rfl⟩
  simpa only [map_update_zero] using this.symm.map hf


/-- If a multilinear map in finitely many variables on seminormed spaces
sends vectors with a component of norm zero to vectors of norm zero
and satisfies the inequality `‖f m‖ ≤ C * ∏ i, ‖m i‖` on a shell `ε i / ‖c i‖ < ‖m i‖ < ε i`
for some positive numbers `ε i` and elements `c i : 𝕜`, `1 < ‖c i‖`,
then it satisfies this inequality for all `m`.

The first assumption is automatically satisfied on normed spaces, see `bound_of_shell` below.
For seminormed spaces, it follows from continuity of `f`, see next lemma,
see `bound_of_shell_of_continuous` below. -/
theorem bound_of_shell_of_norm_map_coord_zero (f : MultilinearMap 𝕜 E G)
    (hf₀ : ∀ {m i}, ‖m i‖ = 0 → ‖f m‖ = 0)
    {ε : ι → ℝ} {C : ℝ} (hε : ∀ i, 0 < ε i) {c : ι → 𝕜} (hc : ∀ i, 1 < ‖c i‖)
    (hf : ∀ m : ∀ i, E i, (∀ i, ε i / ‖c i‖ ≤ ‖m i‖) → (∀ i, ‖m i‖ < ε i) → ‖f m‖ ≤ C * ∏ i, ‖m i‖)
    (m : ∀ i, E i) : ‖f m‖ ≤ C * ∏ i, ‖m i‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf₀ : ∀ {m : (i : ι) → E i} {i : ι}, Eq (Norm.norm (m i)) 0 → Eq (Norm.norm (f …
    ε : ι → Real
    C : Real
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    c : ι → 𝕜
    hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
    hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
    m : (i : ι) → E i
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m …
  -/
  rcases em (∃ i, ‖m i‖ = 0) with (⟨i, hi⟩ | hm)
    /-
      case inl.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : MultilinearMap 𝕜 E G
      hf₀ : ∀ {m : (i : ι) → E i} {i : ι}, Eq (Norm.norm (m i)) 0 → Eq (Norm.norm (f …
      ε : ι → Real
      C : Real
      hε : ∀ (i : ι), LT.lt 0 (ε i)
      c : ι → 𝕜
      hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
      hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
      m : (i : ι) → E i
      i : ι
      hi : Eq (Norm.norm (m i)) 0
      ⊢ LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m …
    -/
  · rw [hf₀ hi, prod_eq_zero (mem_univ i) hi, mul_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf₀ : ∀ {m : (i : ι) → E i} {i : ι}, Eq (Norm.norm (m i)) 0 → Eq (Norm.norm (f …
    ε : ι → Real
    C : Real
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    c : ι → 𝕜
    hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
    hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
    m : (i : ι) → E i
    hm : Not (Exists fun i => Eq (Norm.norm (m i)) 0)
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m …
  -/
  push_neg at hm
  /-
    case inr
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf₀ : ∀ {m : (i : ι) → E i} {i : ι}, Eq (Norm.norm (m i)) 0 → Eq (Norm.norm (f …
    ε : ι → Real
    C : Real
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    c : ι → 𝕜
    hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
    hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
    m : (i : ι) → E i
    hm : ∀ (i : ι), Ne (Norm.norm (m i)) 0
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m …
  -/
  choose δ hδ0 hδm_lt hle_δm _ using fun i => rescale_to_shell_semi_normed (hc i) (hε i) (hm i)
  /-
    case inr
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf₀ : ∀ {m : (i : ι) → E i} {i : ι}, Eq (Norm.norm (m i)) 0 → Eq (Norm.norm (f …
    ε : ι → Real
    C : Real
    hε : ∀ (i : ι), LT.lt 0 (ε i)
    c : ι → 𝕜
    hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
    hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
    m : (i : ι) → E i
    hm : ∀ (i : ι), Ne (Norm.norm (m i)) 0
    δ : ι → 𝕜
    hδ0 : ∀ (i : ι), Ne (δ i) 0
    hδm_lt : ∀ (i : ι), LT.lt (Norm.norm (HSMul.hSMul (δ i) (m i))) (ε i)
    hle_δm : ∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i))) (Norm.norm (HSMu …
    a✝ : ∀ (i : ι), LE.le (Inv.inv (Norm.norm (δ i))) (HMul.hMul (HMul.hMul (Inv.i …
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m …
  -/
  have hδ0 : 0 < ∏ i, ‖δ i‖ := prod_pos fun i _ => norm_pos_iff.2 (hδ0 i)
  simpa [map_smul_univ, norm_smul, prod_mul_distrib, mul_left_comm C, mul_le_mul_left hδ0] using
    hf (fun i => δ i • m i) hle_δm hδm_lt


/-- If a continuous multilinear map in finitely many variables on normed spaces satisfies
the inequality `‖f m‖ ≤ C * ∏ i, ‖m i‖` on a shell `ε i / ‖c i‖ < ‖m i‖ < ε i` for some positive
numbers `ε i` and elements `c i : 𝕜`, `1 < ‖c i‖`, then it satisfies this inequality for all `m`. -/
theorem bound_of_shell_of_continuous (f : MultilinearMap 𝕜 E G) (hfc : Continuous f)
    {ε : ι → ℝ} {C : ℝ} (hε : ∀ i, 0 < ε i) {c : ι → 𝕜} (hc : ∀ i, 1 < ‖c i‖)
    (hf : ∀ m : ∀ i, E i, (∀ i, ε i / ‖c i‖ ≤ ‖m i‖) → (∀ i, ‖m i‖ < ε i) → ‖f m‖ ≤ C * ∏ i, ‖m i‖)
    (m : ∀ i, E i) : ‖f m‖ ≤ C * ∏ i, ‖m i‖ :=
  bound_of_shell_of_norm_map_coord_zero f (norm_map_coord_zero f hfc) hε hc hf m


/-- If a multilinear map in finitely many variables on normed spaces is continuous, then it
satisfies the inequality `‖f m‖ ≤ C * ∏ i, ‖m i‖`, for some `C` which can be chosen to be
positive. -/
theorem exists_bound_of_continuous (f : MultilinearMap 𝕜 E G) (hf : Continuous f) :
    ∃ C : ℝ, 0 < C ∧ ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : MultilinearMap 𝕜 E G
      hf : Continuous ⇑f
      h✝ : IsEmpty ι
      ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
    -/
  · refine ⟨‖f 0‖ + 1, add_pos_of_nonneg_of_pos (norm_nonneg _) zero_lt_one, fun m => ?_⟩
    /-
      case inl
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : MultilinearMap 𝕜 E G
      hf : Continuous ⇑f
      h✝ : IsEmpty ι
      m : (i : ι) → E i
      ⊢ LE.le (Norm.norm (f m)) (HMul.hMul (HAdd.hAdd (Norm.norm (f 0)) 1) (Finset.u …
    -/
    obtain rfl : m = 0 := funext (IsEmpty.elim ‹_›)
    /-
      case inl
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : MultilinearMap 𝕜 E G
      hf : Continuous ⇑f
      h✝ : IsEmpty ι
      ⊢ LE.le (Norm.norm (f 0)) (HMul.hMul (HAdd.hAdd (Norm.norm (f 0)) 1) (Finset.u …
    -/
    simp [univ_eq_empty, zero_le_one]
    /-
      🎉 no goals
    -/
  obtain ⟨ε : ℝ, ε0 : 0 < ε, hε : ∀ m : ∀ i, E i, ‖m - 0‖ < ε → ‖f m - f 0‖ < 1⟩ :=
    NormedAddCommGroup.tendsto_nhds_nhds.1 (hf.tendsto 0) 1 zero_lt_one
  /-
    case inr.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm (HSub.hSub m 0)) ε → LT.lt (Norm. …
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
  -/
  simp only [sub_zero, f.map_zero] at hε
  /-
    case inr.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
  -/
  have : 0 < (‖c‖ / ε) ^ Fintype.card ι := pow_pos (div_pos (zero_lt_one.trans hc) ε0) _
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (m : (i : ι) → E i), LE.le (Norm.norm (f  …
  -/
  refine ⟨_, this, ?_⟩
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))
    ⊢ ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul (HPow.hPow (HDiv.h …
  -/
  refine f.bound_of_shell_of_continuous hf (fun _ => ε0) (fun _ => hc) fun m hcm hm => ?_
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))
    m : (i : ι) → E i
    hcm : ∀ (i : ι), LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm (m i))
    hm : ∀ (i : ι), LT.lt (Norm.norm (m i)) ε
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (F …
  -/
  refine (hε m ((pi_norm_lt_iff ε0).2 hm)).le.trans ?_
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))
    m : (i : ι) → E i
    hcm : ∀ (i : ι), LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm (m i))
    hm : ∀ (i : ι), LT.lt (Norm.norm (m i)) ε
    ⊢ LE.le 1 (HMul.hMul (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))  …
  -/
  rw [← div_le_iff₀' this, one_div, ← inv_pow, inv_div, Fintype.card, ← prod_const]
  /-
    case inr.intro.intro.intro
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    hf : Continuous ⇑f
    h✝ : Nonempty ι
    ε : Real
    ε0 : LT.lt 0 ε
    hε : ∀ (m : (i : ι) → E i), LT.lt (Norm.norm m) ε → LT.lt (Norm.norm (f m)) 1
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HPow.hPow (HDiv.hDiv (Norm.norm c) ε) (Fintype.card ι))
    m : (i : ι) → E i
    hcm : ∀ (i : ι), LE.le (HDiv.hDiv ε (Norm.norm c)) (Norm.norm (m i))
    hm : ∀ (i : ι), LT.lt (Norm.norm (m i)) ε
    ⊢ LE.le (Finset.univ.prod fun _x => HDiv.hDiv ε (Norm.norm c)) (Finset.univ.pr …
  -/
  exact prod_le_prod (fun _ _ => div_nonneg ε0.le (norm_nonneg _)) fun i _ => hcm i
  /-
    🎉 no goals
  -/


/-- If a multilinear map `f` satisfies a boundedness property around `0`,
one can deduce a bound on `f m₁ - f m₂` using the multilinearity.
Here, we give a precise but hard to use version.
See `norm_image_sub_le_of_bound` for a less precise but more usable version.
The bound reads
`‖f m - f m'‖ ≤
  C * ‖m 1 - m' 1‖ * max ‖m 2‖ ‖m' 2‖ * max ‖m 3‖ ‖m' 3‖ * ... * max ‖m n‖ ‖m' n‖ + ...`,
where the other terms in the sum are the same products where `1` is replaced by any `i`. -/
theorem norm_image_sub_le_of_bound' [DecidableEq ι] (f : MultilinearMap 𝕜 E G) {C : ℝ} (hC : 0 ≤ C)
    (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) (m₁ m₂ : ∀ i, E i) :
    ‖f m₁ - f m₂‖ ≤ C * ∑ i, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ := by
  have A :
    ∀ s : Finset ι,
      ‖f m₁ - f (s.piecewise m₂ m₁)‖ ≤
        C * ∑ i ∈ s, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ := by
    intro s
    induction' s using Finset.induction with i s his Hrec
    · simp
    have I :
      ‖f (s.piecewise m₂ m₁) - f ((insert i s).piecewise m₂ m₁)‖ ≤
        C * ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ := by
      have A : (insert i s).piecewise m₂ m₁ = Function.update (s.piecewise m₂ m₁) i (m₂ i) :=
        s.piecewise_insert _ _ _
      have B : s.piecewise m₂ m₁ = Function.update (s.piecewise m₂ m₁) i (m₁ i) := by
        simp [eq_update_iff, his]
      rw [B, A, ← f.map_update_sub]
      apply le_trans (H _)
      gcongr with j
      by_cases h : j = i
      · rw [h]
        simp
      · by_cases h' : j ∈ s <;> simp [h', h, le_refl]
    calc
      ‖f m₁ - f ((insert i s).piecewise m₂ m₁)‖ ≤
          ‖f m₁ - f (s.piecewise m₂ m₁)‖ +
            ‖f (s.piecewise m₂ m₁) - f ((insert i s).piecewise m₂ m₁)‖ := by
        rw [← dist_eq_norm, ← dist_eq_norm, ← dist_eq_norm]
        exact dist_triangle _ _ _
      _ ≤ (C * ∑ i ∈ s, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖) +
            C * ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ :=
        (add_le_add Hrec I)
      _ = C * ∑ i ∈ insert i s, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ := by
        simp [his, add_comm, left_distrib]
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : MultilinearMap 𝕜 E G
    C : Real
    hC : LE.le 0 C
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.p …
    m₁ m₂ : (i : ι) → E i
    A : ∀ (s : Finset ι), LE.le (Norm.norm (HSub.hSub (f m₁) (f (s.piecewise m₂ m₁ …
    ⊢ LE.le (Norm.norm (HSub.hSub (f m₁) (f m₂))) (HMul.hMul C (Finset.univ.sum fu …
  -/
  convert A univ
  /-
    case h.e'_3.h.e'_3.h.e'_6.h.e'_6
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    f : MultilinearMap 𝕜 E G
    C : Real
    hC : LE.le 0 C
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.p …
    m₁ m₂ : (i : ι) → E i
    A : ∀ (s : Finset ι), LE.le (Norm.norm (HSub.hSub (f m₁) (f (s.piecewise m₂ m₁ …
    ⊢ Eq m₂ (Finset.univ.piecewise m₂ m₁)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `f` satisfies a boundedness property around `0`, one can deduce a bound on `f m₁ - f m₂`
using the multilinearity. Here, we give a usable but not very precise version. See
`norm_image_sub_le_of_bound'` for a more precise but less usable version. The bound is
`‖f m - f m'‖ ≤ C * card ι * ‖m - m'‖ * (max ‖m‖ ‖m'‖) ^ (card ι - 1)`. -/
theorem norm_image_sub_le_of_bound (f : MultilinearMap 𝕜 E G)
    {C : ℝ} (hC : 0 ≤ C) (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) (m₁ m₂ : ∀ i, E i) :
    ‖f m₁ - f m₂‖ ≤ C * Fintype.card ι * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) * ‖m₁ - m₂‖ := by
  classical
  have A :
    ∀ i : ι,
      ∏ j, (if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖) ≤
        ‖m₁ - m₂‖ * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) := by
    intro i
    calc
      ∏ j, (if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖) ≤
          ∏ j : ι, Function.update (fun _ => max ‖m₁‖ ‖m₂‖) i ‖m₁ - m₂‖ j := by
        apply Finset.prod_le_prod
        · intro j _
          by_cases h : j = i <;> simp [h, norm_nonneg]
        · intro j _
          by_cases h : j = i
          · rw [h]
            simp only [ite_true, Function.update_self]
            exact norm_le_pi_norm (m₁ - m₂) i
          · simp [h, - le_sup_iff, - sup_le_iff, sup_le_sup, norm_le_pi_norm]
      _ = ‖m₁ - m₂‖ * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) := by
        rw [prod_update_of_mem (Finset.mem_univ _)]
        simp [card_univ_diff]
  calc
    ‖f m₁ - f m₂‖ ≤ C * ∑ i, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ :=
      f.norm_image_sub_le_of_bound' hC H m₁ m₂
    _ ≤ C * ∑ _i, ‖m₁ - m₂‖ * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) := by gcongr; apply A
    _ = C * Fintype.card ι * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) * ‖m₁ - m₂‖ := by
      rw [sum_const, card_univ, nsmul_eq_mul]
      ring


/-- If a multilinear map satisfies an inequality `‖f m‖ ≤ C * ∏ i, ‖m i‖`, then it is
continuous. -/
theorem continuous_of_bound (f : MultilinearMap 𝕜 E G) (C : ℝ) (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) :
    Continuous f := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    C : Real
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.p …
    ⊢ Continuous ⇑f
  -/
  let D := max C 1
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    C : Real
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.p …
    D : Real := Max.max C 1
    ⊢ Continuous ⇑f
  -/
  have D_pos : 0 ≤ D := le_trans zero_le_one (le_max_right _ _)
  replace H (m) : ‖f m‖ ≤ D * ∏ i, ‖m i‖ :=
    (H m).trans (mul_le_mul_of_nonneg_right (le_max_left _ _) <| by positivity)
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    C : Real
    D : Real := Max.max C 1
    D_pos : LE.le 0 D
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul D (Finset.univ.p …
    ⊢ Continuous ⇑f
  -/
  refine continuous_iff_continuousAt.2 fun m => ?_
  refine
    continuousAt_of_locally_lipschitz zero_lt_one
      (D * Fintype.card ι * (‖m‖ + 1) ^ (Fintype.card ι - 1)) fun m' h' => ?_
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : MultilinearMap 𝕜 E G
    C : Real
    D : Real := Max.max C 1
    D_pos : LE.le 0 D
    H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul D (Finset.univ.p …
    m m' : (i : ι) → E i
    h' : LT.lt (Dist.dist m' m) 1
    ⊢ LE.le (Dist.dist (f m') (f m)) (HMul.hMul (HMul.hMul (HMul.hMul D ↑(Fintype. …
  -/
  rw [dist_eq_norm, dist_eq_norm]
  have : max ‖m'‖ ‖m‖ ≤ ‖m‖ + 1 := by
    simp [zero_le_one, norm_le_of_mem_closedBall (le_of_lt h')]
  calc
    ‖f m' - f m‖ ≤ D * Fintype.card ι * max ‖m'‖ ‖m‖ ^ (Fintype.card ι - 1) * ‖m' - m‖ :=
      f.norm_image_sub_le_of_bound D_pos H m' m
    _ ≤ D * Fintype.card ι * (‖m‖ + 1) ^ (Fintype.card ι - 1) * ‖m' - m‖ := by gcongr


/-- Constructing a continuous multilinear map from a multilinear map satisfying a boundedness
condition. -/
def mkContinuous (f : MultilinearMap 𝕜 E G) (C : ℝ) (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) :
    ContinuousMultilinearMap 𝕜 E G :=
  { f with cont := f.continuous_of_bound C H }


@[simp]
theorem coe_mkContinuous (f : MultilinearMap 𝕜 E G) (C : ℝ) (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) :
    ⇑(f.mkContinuous C H) = f :=
  rfl


/-- Given a multilinear map in `n` variables, if one restricts it to `k` variables putting `z` on
the other coordinates, then the resulting restricted function satisfies an inequality
`‖f.restr v‖ ≤ C * ‖z‖^(n-k) * Π ‖v i‖` if the original function satisfies `‖f v‖ ≤ C * Π ‖v i‖`. -/
theorem restr_norm_le {k n : ℕ} (f : MultilinearMap 𝕜 (fun _ : Fin n => G) G')
    (s : Finset (Fin n)) (hk : #s = k) (z : G) {C : ℝ} (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖)
    (v : Fin k → G) : ‖f.restr s hk z v‖ ≤ C * ‖z‖ ^ (n - k) * ∏ i, ‖v i‖ := by
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : SeminormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k n : Nat
    f : MultilinearMap 𝕜 (fun x => G) G'
    s : Finset (Fin n)
    hk : Eq s.card k
    z : G
    C : Real
    H : ∀ (m : Fin n → G), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod  …
    v : Fin k → G
    ⊢ LE.le (Norm.norm ((f.restr s hk z) v)) (HMul.hMul (HMul.hMul C (HPow.hPow (N …
  -/
  rw [mul_right_comm, mul_assoc]
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : SeminormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k n : Nat
    f : MultilinearMap 𝕜 (fun x => G) G'
    s : Finset (Fin n)
    hk : Eq s.card k
    z : G
    C : Real
    H : ∀ (m : Fin n → G), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod  …
    v : Fin k → G
    ⊢ LE.le (Norm.norm ((f.restr s hk z) v)) (HMul.hMul C (HMul.hMul (Finset.univ. …
  -/
  convert H _ using 2
  simp only [apply_dite norm, Fintype.prod_dite, prod_const ‖z‖, Finset.card_univ,
    Fintype.card_of_subtype sᶜ fun _ => mem_compl, card_compl, Fintype.card_fin, hk, mk_coe, ←
    (s.orderIsoOfFin hk).symm.bijective.prod_comp fun x => ‖v x‖]
  /-
    case h.e'_4.h.e'_6
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : SeminormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k n : Nat
    f : MultilinearMap 𝕜 (fun x => G) G'
    s : Finset (Fin n)
    hk : Eq s.card k
    z : G
    C : Real
    H : ∀ (m : Fin n → G), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.prod  …
    v : Fin k → G
    ⊢ Eq (HMul.hMul (Finset.univ.prod fun i => Norm.norm (v ((s.orderIsoOfFin hk). …
  -/
  convert rfl
  /-
    🎉 no goals
  -/


theorem bound (f : ContinuousMultilinearMap 𝕜 E G) :
    ∃ C : ℝ, 0 < C ∧ ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖ :=
  f.toMultilinearMap.exists_bound_of_continuous f.2


/-- The operator norm of a continuous multilinear map is the inf of all its bounds. -/
def opNorm (f : ContinuousMultilinearMap 𝕜 E G) : ℝ :=
  sInf { c | 0 ≤ (c : ℝ) ∧ ∀ m, ‖f m‖ ≤ c * ∏ i, ‖m i‖ }


instance hasOpNorm : Norm (ContinuousMultilinearMap 𝕜 E G) :=
  ⟨opNorm⟩


/-- An alias of `ContinuousMultilinearMap.hasOpNorm` with non-dependent types to help typeclass
search. -/
instance hasOpNorm' : Norm (ContinuousMultilinearMap 𝕜 (fun _ : ι => G) G') :=
  ContinuousMultilinearMap.hasOpNorm


theorem norm_def (f : ContinuousMultilinearMap 𝕜 E G) :
    ‖f‖ = sInf { c | 0 ≤ (c : ℝ) ∧ ∀ m, ‖f m‖ ≤ c * ∏ i, ‖m i‖ } :=
  rfl

-- So that invocations of `le_csInf` make sense: we show that the set of
-- bounds is nonempty and bounded below.

theorem bounds_nonempty {f : ContinuousMultilinearMap 𝕜 E G} :
    ∃ c, c ∈ { c | 0 ≤ c ∧ ∀ m, ‖f m‖ ≤ c * ∏ i, ‖m i‖ } :=
  let ⟨M, hMp, hMb⟩ := f.bound
  ⟨M, le_of_lt hMp, hMb⟩


theorem bounds_bddBelow {f : ContinuousMultilinearMap 𝕜 E G} :
    BddBelow { c | 0 ≤ c ∧ ∀ m, ‖f m‖ ≤ c * ∏ i, ‖m i‖ } :=
  ⟨0, fun _ ⟨hn, _⟩ => hn⟩


theorem isLeast_opNorm (f : ContinuousMultilinearMap 𝕜 E G) :
    IsLeast {c : ℝ | 0 ≤ c ∧ ∀ m, ‖f m‖ ≤ c * ∏ i, ‖m i‖} ‖f‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E G
    ⊢ IsLeast (setOf fun c => And (LE.le 0 c) (∀ (m : (i : ι) → E i), LE.le (Norm. …
  -/
  refine IsClosed.isLeast_csInf ?_ bounds_nonempty bounds_bddBelow
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E G
    ⊢ IsClosed (setOf fun c => And (LE.le 0 c) (∀ (m : (i : ι) → E i), LE.le (Norm …
  -/
  simp only [Set.setOf_and, Set.setOf_forall]
  exact isClosed_Ici.inter (isClosed_iInter fun m ↦
    isClosed_le continuous_const (continuous_id.mul continuous_const))


@[deprecated (since := "2024-02-02")] alias isLeast_op_norm := isLeast_opNorm


theorem opNorm_nonneg (f : ContinuousMultilinearMap 𝕜 E G) : 0 ≤ ‖f‖ :=
  Real.sInf_nonneg fun _ ⟨hx, _⟩ => hx


@[deprecated (since := "2024-02-02")] alias op_norm_nonneg := opNorm_nonneg


/-- The fundamental property of the operator norm of a continuous multilinear map:
`‖f m‖` is bounded by `‖f‖` times the product of the `‖m i‖`. -/
theorem le_opNorm (f : ContinuousMultilinearMap 𝕜 E G) (m : ∀ i, E i) :
    ‖f m‖ ≤ ‖f‖ * ∏ i, ‖m i‖ :=
  f.isLeast_opNorm.1.2 m


@[deprecated (since := "2024-02-02")] alias le_op_norm := le_opNorm


theorem le_mul_prod_of_opNorm_le_of_le {f : ContinuousMultilinearMap 𝕜 E G}
    {m : ∀ i, E i} {C : ℝ} {b : ι → ℝ} (hC : ‖f‖ ≤ C) (hm : ∀ i, ‖m i‖ ≤ b i) :
    ‖f m‖ ≤ C * ∏ i, b i :=
                              /-
                                𝕜 : Type u
                                ι : Type v
                                E : ι → Type wE
                                G : Type wG
                                inst✝⁵ : NontriviallyNormedField 𝕜
                                inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                                inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                                inst✝² : SeminormedAddCommGroup G
                                inst✝¹ : NormedSpace 𝕜 G
                                inst✝ : Fintype ι
                                f : ContinuousMultilinearMap 𝕜 E G
                                m : (i : ι) → E i
                                C : Real
                                b : ι → Real
                                hC : LE.le (Norm.norm f) C
                                hm : ∀ (i : ι), LE.le (Norm.norm (m i)) (b i)
                                ⊢ LE.le (HMul.hMul (Norm.norm f) (Finset.univ.prod fun i => Norm.norm (m i)))  …
                              -/
  (f.le_opNorm m).trans <| by gcongr; exacts [f.opNorm_nonneg.trans hC, hm _]
                                      /-
                                        🎉 no goals
                                      -/


@[deprecated (since := "2024-02-02")]
alias le_mul_prod_of_le_op_norm_of_le := le_mul_prod_of_opNorm_le_of_le


@[deprecated (since := "2024-11-27")]
alias le_mul_prod_of_le_opNorm_of_le := le_mul_prod_of_opNorm_le_of_le


theorem le_opNorm_mul_prod_of_le (f : ContinuousMultilinearMap 𝕜 E G)
    {m : ∀ i, E i} {b : ι → ℝ} (hm : ∀ i, ‖m i‖ ≤ b i) : ‖f m‖ ≤ ‖f‖ * ∏ i, b i :=
  le_mul_prod_of_opNorm_le_of_le le_rfl hm


@[deprecated (since := "2024-02-02")] alias le_op_norm_mul_prod_of_le := le_opNorm_mul_prod_of_le


theorem le_opNorm_mul_pow_card_of_le (f : ContinuousMultilinearMap 𝕜 E G) {m b} (hm : ‖m‖ ≤ b) :
    ‖f m‖ ≤ ‖f‖ * b ^ Fintype.card ι := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E G
    m : (i : ι) → E i
    b : Real
    hm : LE.le (Norm.norm m) b
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul (Norm.norm f) (HPow.hPow b (Fintype.card  …
  -/
  simpa only [prod_const] using f.le_opNorm_mul_prod_of_le fun i => (norm_le_pi_norm m i).trans hm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")]
alias le_op_norm_mul_pow_card_of_le := le_opNorm_mul_pow_card_of_le


theorem le_opNorm_mul_pow_of_le {n : ℕ} {Ei : Fin n → Type*} [∀ i, SeminormedAddCommGroup (Ei i)]
    [∀ i, NormedSpace 𝕜 (Ei i)] (f : ContinuousMultilinearMap 𝕜 Ei G) {m : ∀ i, Ei i} {b : ℝ}
    (hm : ‖m‖ ≤ b) : ‖f m‖ ≤ ‖f‖ * b ^ n := by
  /-
    𝕜 : Type u
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    n : Nat
    Ei : Fin n → Type u_1
    inst✝¹ : (i : Fin n) → SeminormedAddCommGroup (Ei i)
    inst✝ : (i : Fin n) → NormedSpace 𝕜 (Ei i)
    f : ContinuousMultilinearMap 𝕜 Ei G
    m : (i : Fin n) → Ei i
    b : Real
    hm : LE.le (Norm.norm m) b
    ⊢ LE.le (Norm.norm (f m)) (HMul.hMul (Norm.norm f) (HPow.hPow b n))
  -/
  simpa only [Fintype.card_fin] using f.le_opNorm_mul_pow_card_of_le hm
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias le_op_norm_mul_pow_of_le := le_opNorm_mul_pow_of_le


theorem le_of_opNorm_le {f : ContinuousMultilinearMap 𝕜 E G} {C : ℝ} (h : ‖f‖ ≤ C) (m : ∀ i, E i) :
    ‖f m‖ ≤ C * ∏ i, ‖m i‖ :=
  le_mul_prod_of_opNorm_le_of_le h fun _ ↦ le_rfl


@[deprecated (since := "2024-02-02")] alias le_of_op_norm_le := le_of_opNorm_le


theorem ratio_le_opNorm (f : ContinuousMultilinearMap 𝕜 E G) (m : ∀ i, E i) :
    (‖f m‖ / ∏ i, ‖m i‖) ≤ ‖f‖ :=
                        /-
                          𝕜 : Type u
                          ι : Type v
                          E : ι → Type wE
                          G : Type wG
                          inst✝⁵ : NontriviallyNormedField 𝕜
                          inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                          inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                          inst✝² : SeminormedAddCommGroup G
                          inst✝¹ : NormedSpace 𝕜 G
                          inst✝ : Fintype ι
                          f : ContinuousMultilinearMap 𝕜 E G
                          m : (i : ι) → E i
                          ⊢ LE.le 0 (Finset.univ.prod fun i => Norm.norm (m i))
                        -/
  div_le_of_le_mul₀ (by positivity) (opNorm_nonneg _) (f.le_opNorm m)
                        /-
                          🎉 no goals
                        -/


@[deprecated (since := "2024-02-02")] alias ratio_le_op_norm := ratio_le_opNorm


/-- The image of the unit ball under a continuous multilinear map is bounded. -/
theorem unit_le_opNorm (f : ContinuousMultilinearMap 𝕜 E G) {m : ∀ i, E i} (h : ‖m‖ ≤ 1) :
    ‖f m‖ ≤ ‖f‖ :=
                                                 /-
                                                   𝕜 : Type u
                                                   ι : Type v
                                                   E : ι → Type wE
                                                   G : Type wG
                                                   inst✝⁵ : NontriviallyNormedField 𝕜
                                                   inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                                                   inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                                                   inst✝² : SeminormedAddCommGroup G
                                                   inst✝¹ : NormedSpace 𝕜 G
                                                   inst✝ : Fintype ι
                                                   f : ContinuousMultilinearMap 𝕜 E G
                                                   m : (i : ι) → E i
                                                   h : LE.le (Norm.norm m) 1
                                                   ⊢ LE.le (HMul.hMul (Norm.norm f) (HPow.hPow 1 (Fintype.card ι))) (Norm.norm f)
                                                 -/
  (le_opNorm_mul_pow_card_of_le f h).trans <| by simp
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-02-02")] alias unit_le_op_norm := unit_le_opNorm


/-- If one controls the norm of every `f x`, then one controls the norm of `f`. -/
theorem opNorm_le_bound {f : ContinuousMultilinearMap 𝕜 E G}
    {M : ℝ} (hMp : 0 ≤ M) (hM : ∀ m, ‖f m‖ ≤ M * ∏ i, ‖m i‖) : ‖f‖ ≤ M :=
  csInf_le bounds_bddBelow ⟨hMp, hM⟩


@[deprecated (since := "2024-02-02")] alias op_norm_le_bound := opNorm_le_bound


theorem opNorm_le_iff {f : ContinuousMultilinearMap 𝕜 E G} {C : ℝ} (hC : 0 ≤ C) :
    ‖f‖ ≤ C ↔ ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖ :=
  ⟨fun h _ ↦ le_of_opNorm_le h _, opNorm_le_bound hC⟩


@[deprecated (since := "2024-02-02")] alias op_norm_le_iff := opNorm_le_iff


/-- The operator norm satisfies the triangle inequality. -/
theorem opNorm_add_le (f g : ContinuousMultilinearMap 𝕜 E G) : ‖f + g‖ ≤ ‖f‖ + ‖g‖ :=
  opNorm_le_bound (add_nonneg (opNorm_nonneg f) (opNorm_nonneg g)) fun x => by
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f g : ContinuousMultilinearMap 𝕜 E G
      x : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((HAdd.hAdd f g) x)) (HMul.hMul (HAdd.hAdd (Norm.norm f) (N …
    -/
    rw [add_mul]
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f g : ContinuousMultilinearMap 𝕜 E G
      x : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((HAdd.hAdd f g) x)) (HAdd.hAdd (HMul.hMul (Norm.norm f) (F …
    -/
    exact norm_add_le_of_le (le_opNorm _ _) (le_opNorm _ _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_norm_add_le := opNorm_add_le


theorem opNorm_zero : ‖(0 : ContinuousMultilinearMap 𝕜 E G)‖ = 0 :=
                                                                    /-
                                                                      𝕜 : Type u
                                                                      ι : Type v
                                                                      E : ι → Type wE
                                                                      G : Type wG
                                                                      inst✝⁵ : NontriviallyNormedField 𝕜
                                                                      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                                                                      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                                                                      inst✝² : SeminormedAddCommGroup G
                                                                      inst✝¹ : NormedSpace 𝕜 G
                                                                      inst✝ : Fintype ι
                                                                      m : (i : ι) → E i
                                                                      ⊢ LE.le (Norm.norm (0 m)) (HMul.hMul 0 (Finset.univ.prod fun i => Norm.norm (m …
                                                                    -/
  (opNorm_nonneg _).antisymm' <| opNorm_le_bound le_rfl fun m => by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[deprecated (since := "2024-02-02")] alias op_norm_zero := opNorm_zero


theorem opNorm_smul_le (c : 𝕜') (f : ContinuousMultilinearMap 𝕜 E G) : ‖c • f‖ ≤ ‖c‖ * ‖f‖ :=
  (c • f).opNorm_le_bound (mul_nonneg (norm_nonneg _) (opNorm_nonneg _)) fun m ↦ by
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : Fintype ι
      𝕜' : Type u_1
      inst✝² : NormedField 𝕜'
      inst✝¹ : NormedSpace 𝕜' G
      inst✝ : SMulCommClass 𝕜 𝕜' G
      c : 𝕜'
      f : ContinuousMultilinearMap 𝕜 E G
      m : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((HSMul.hSMul c f) m)) (HMul.hMul (HMul.hMul (Norm.norm c)  …
    -/
    rw [smul_apply, norm_smul, mul_assoc]
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁸ : NontriviallyNormedField 𝕜
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : Fintype ι
      𝕜' : Type u_1
      inst✝² : NormedField 𝕜'
      inst✝¹ : NormedSpace 𝕜' G
      inst✝ : SMulCommClass 𝕜 𝕜' G
      c : 𝕜'
      f : ContinuousMultilinearMap 𝕜 E G
      m : (i : ι) → E i
      ⊢ LE.le (HMul.hMul (Norm.norm c) (Norm.norm (f m))) (HMul.hMul (Norm.norm c) ( …
    -/
    exact mul_le_mul_of_nonneg_left (le_opNorm _ _) (norm_nonneg _)
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_norm_smul_le := opNorm_smul_le


variable (𝕜 E G) in
/-- Operator seminorm on the space of continuous multilinear maps, as `Seminorm`.

We use this seminorm
to define a `SeminormedAddCommGroup` structure on `ContinuousMultilinearMap 𝕜 E G`,
but we have to override the projection `UniformSpace`
so that it is definitionally equal to the one coming from the topologies on `E` and `G`. -/
protected def seminorm : Seminorm 𝕜 (ContinuousMultilinearMap 𝕜 E G) :=
  .ofSMulLE norm opNorm_zero opNorm_add_le fun c f ↦ f.opNorm_smul_le c


private lemma uniformity_eq_seminorm :
    𝓤 (ContinuousMultilinearMap 𝕜 E G) = ⨅ r > 0, 𝓟 {f | ‖f.1 - f.2‖ < r} := by
  refine (ContinuousMultilinearMap.seminorm 𝕜 E G).uniformity_eq_of_hasBasis
    (ContinuousMultilinearMap.hasBasis_nhds_zero_of_basis Metric.nhds_basis_closedBall)
    ?_ fun (s, r) ⟨hs, hr⟩ ↦ ?_
    /-
      case refine_1
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      ⊢ Exists fun r => Membership.mem (nhds 0) ((ContinuousMultilinearMap.seminorm  …
    -/
  · rcases NormedField.exists_lt_norm 𝕜 1 with ⟨c, hc⟩
    /-
      case refine_1.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      ⊢ Exists fun r => Membership.mem (nhds 0) ((ContinuousMultilinearMap.seminorm  …
    -/
    have hc₀ : 0 < ‖c‖ := one_pos.trans hc
    /-
      case refine_1.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      hc₀ : LT.lt 0 (Norm.norm c)
      ⊢ Exists fun r => Membership.mem (nhds 0) ((ContinuousMultilinearMap.seminorm  …
    -/
    simp only [hasBasis_nhds_zero.mem_iff, Prod.exists]
    /-
      case refine_1.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      hc₀ : LT.lt 0 (Norm.norm c)
      ⊢ Exists fun r => Exists fun a => Exists fun b => And (And (Bornology.IsVonNBo …
    -/
    use 1, closedBall 0 ‖c‖, closedBall 0 1
    suffices ∀ f : ContinuousMultilinearMap 𝕜 E G, (∀ x, ‖x‖ ≤ ‖c‖ → ‖f x‖ ≤ 1) → ‖f‖ ≤ 1 by
      simpa [NormedSpace.isVonNBounded_closedBall, closedBall_mem_nhds, Set.subset_def, Set.MapsTo]
    /-
      case h
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      hc₀ : LT.lt 0 (Norm.norm c)
      ⊢ ∀ (f : ContinuousMultilinearMap 𝕜 E G), (∀ (x : (i : ι) → E i), LE.le (Norm. …
    -/
    intro f hf
    refine opNorm_le_bound (by positivity) <|
      f.1.bound_of_shell_of_continuous f.2 (fun _ ↦ hc₀) (fun _ ↦ hc) fun x hcx hx ↦ ?_
    calc
      ‖f x‖ ≤ 1 := hf _ <| (pi_norm_le_iff_of_nonneg (norm_nonneg c)).2 fun i ↦ (hx i).le
      _ = ∏ i : ι, 1 := by simp
      _ ≤ ∏ i, ‖x i‖ := Finset.prod_le_prod (fun _ _ ↦ zero_le_one) fun i _ ↦ by
        simpa only [div_self hc₀.ne'] using hcx i
      _ = 1 * ∏ i, ‖x i‖ := (one_mul _).symm
    /-
      case refine_2
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset ((ContinuousMultilinea …
    -/
  · rcases (NormedSpace.isVonNBounded_iff' _).1 hs with ⟨ε, hε⟩
    /-
      case refine_2.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : (i : ι) → E i), Membership.mem { fst := s, snd := r }.1 x → LE.le  …
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset ((ContinuousMultilinea …
    -/
    rcases exists_pos_mul_lt hr (ε ^ Fintype.card ι) with ⟨δ, hδ₀, hδ⟩
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : (i : ι) → E i), Membership.mem { fst := s, snd := r }.1 x → LE.le  …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul (HPow.hPow ε (Fintype.card ι)) δ) { fst := s, snd := r }.2
      ⊢ Exists fun r_1 => And (GT.gt r_1 0) (HasSubset.Subset ((ContinuousMultilinea …
    -/
    refine ⟨δ, hδ₀, fun f hf x hx ↦ ?_⟩
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : (i : ι) → E i), Membership.mem { fst := s, snd := r }.1 x → LE.le  …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul (HPow.hPow ε (Fintype.card ι)) δ) { fst := s, snd := r }.2
      f : ContinuousMultilinearMap 𝕜 E G
      hf : Membership.mem ((ContinuousMultilinearMap.seminorm 𝕜 E G).ball 0 δ) f
      x : (i : ι) → E i
      hx : Membership.mem { fst := s, snd := r }.1 x
      ⊢ Membership.mem (Metric.closedBall 0 { fst := s, snd := r }.2) (f x)
    -/
    simp only [Seminorm.mem_ball_zero, mem_closedBall_zero_iff] at hf ⊢
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : (i : ι) → E i), Membership.mem { fst := s, snd := r }.1 x → LE.le  …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul (HPow.hPow ε (Fintype.card ι)) δ) { fst := s, snd := r }.2
      f : ContinuousMultilinearMap 𝕜 E G
      x : (i : ι) → E i
      hx : Membership.mem { fst := s, snd := r }.1 x
      hf : LT.lt ((ContinuousMultilinearMap.seminorm 𝕜 E G) f) δ
      ⊢ LE.le (Norm.norm (f x)) r
    -/
    replace hf : ‖f‖ ≤ δ := hf.le
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      x✝¹ : Prod (Set ((i : ι) → E i)) Real
      s : Set ((i : ι) → E i)
      r : Real
      x✝ : And (Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1) (LT.lt 0 { fst : …
      hs : Bornology.IsVonNBounded 𝕜 { fst := s, snd := r }.1
      hr : LT.lt 0 { fst := s, snd := r }.2
      ε : Real
      hε : ∀ (x : (i : ι) → E i), Membership.mem { fst := s, snd := r }.1 x → LE.le  …
      δ : Real
      hδ₀ : LT.lt 0 δ
      hδ : LT.lt (HMul.hMul (HPow.hPow ε (Fintype.card ι)) δ) { fst := s, snd := r }.2
      f : ContinuousMultilinearMap 𝕜 E G
      x : (i : ι) → E i
      hx : Membership.mem { fst := s, snd := r }.1 x
      hf : LE.le (Norm.norm f) δ
      ⊢ LE.le (Norm.norm (f x)) r
    -/
    replace hx : ‖x‖ ≤ ε := hε x hx
    calc
      ‖f x‖ ≤ ‖f‖ * ε ^ Fintype.card ι := le_opNorm_mul_pow_card_of_le f hx
      _ ≤ δ * ε ^ Fintype.card ι := by have := (norm_nonneg x).trans hx; gcongr
      _ ≤ r := (mul_comm _ _).trans_le hδ.le


instance instPseudoMetricSpace : PseudoMetricSpace (ContinuousMultilinearMap 𝕜 E G) :=
  .replaceUniformity
    (ContinuousMultilinearMap.seminorm 𝕜 E G).toSeminormedAddCommGroup.toPseudoMetricSpace
    uniformity_eq_seminorm


/-- Continuous multilinear maps themselves form a seminormed space with respect to
    the operator norm. -/
instance seminormedAddCommGroup :
    SeminormedAddCommGroup (ContinuousMultilinearMap 𝕜 E G) := ⟨fun _ _ ↦ rfl⟩


/-- An alias of `ContinuousMultilinearMap.seminormedAddCommGroup` with non-dependent types to help
typeclass search. -/
instance seminormedAddCommGroup' :
    SeminormedAddCommGroup (ContinuousMultilinearMap 𝕜 (fun _ : ι => G) G') :=
  ContinuousMultilinearMap.seminormedAddCommGroup


instance normedSpace : NormedSpace 𝕜' (ContinuousMultilinearMap 𝕜 E G) :=
  ⟨fun c f => f.opNorm_smul_le c⟩


/-- An alias of `ContinuousMultilinearMap.normedSpace` with non-dependent types to help typeclass
search. -/
instance normedSpace' : NormedSpace 𝕜' (ContinuousMultilinearMap 𝕜 (fun _ : ι => G') G) :=
  ContinuousMultilinearMap.normedSpace


@[deprecated norm_neg (since := "2024-11-24")]
theorem opNorm_neg (f : ContinuousMultilinearMap 𝕜 E G) : ‖-f‖ = ‖f‖ := norm_neg f


@[deprecated (since := "2024-02-02")] alias op_norm_neg := norm_neg


/-- The fundamental property of the operator norm of a continuous multilinear map:
`‖f m‖` is bounded by `‖f‖` times the product of the `‖m i‖`, `nnnorm` version. -/
theorem le_opNNNorm (f : ContinuousMultilinearMap 𝕜 E G) (m : ∀ i, E i) :
    ‖f m‖₊ ≤ ‖f‖₊ * ∏ i, ‖m i‖₊ :=
  NNReal.coe_le_coe.1 <| by
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E G
      m : (i : ι) → E i
      ⊢ LE.le ↑(NNNorm.nnnorm (f m)) ↑(HMul.hMul (NNNorm.nnnorm f) (Finset.univ.prod …
    -/
    push_cast
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E G
      m : (i : ι) → E i
      ⊢ LE.le (Norm.norm (f m)) (HMul.hMul (Norm.norm f) (Finset.univ.prod fun x =>  …
    -/
    exact f.le_opNorm m
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias le_op_nnnorm := le_opNNNorm


theorem le_of_opNNNorm_le (f : ContinuousMultilinearMap 𝕜 E G)
    {C : ℝ≥0} (h : ‖f‖₊ ≤ C) (m : ∀ i, E i) : ‖f m‖₊ ≤ C * ∏ i, ‖m i‖₊ :=
  (f.le_opNNNorm m).trans <| mul_le_mul' h le_rfl


@[deprecated (since := "2024-02-02")] alias le_of_op_nnnorm_le := le_of_opNNNorm_le


theorem opNNNorm_le_iff {f : ContinuousMultilinearMap 𝕜 E G} {C : ℝ≥0} :
    ‖f‖₊ ≤ C ↔ ∀ m, ‖f m‖₊ ≤ C * ∏ i, ‖m i‖₊ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E G
    C : NNReal
    ⊢ Iff (LE.le (NNNorm.nnnorm f) C) (∀ (m : (i : ι) → E i), LE.le (NNNorm.nnnorm …
  -/
  simp only [← NNReal.coe_le_coe]; simp [opNorm_le_iff C.coe_nonneg, NNReal.coe_prod]
                                   /-
                                     🎉 no goals
                                   -/


@[deprecated (since := "2024-02-02")] alias op_nnnorm_le_iff := opNNNorm_le_iff


theorem isLeast_opNNNorm (f : ContinuousMultilinearMap 𝕜 E G) :
    IsLeast {C : ℝ≥0 | ∀ m, ‖f m‖₊ ≤ C * ∏ i, ‖m i‖₊} ‖f‖₊ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E G
    ⊢ IsLeast (setOf fun C => ∀ (m : (i : ι) → E i), LE.le (NNNorm.nnnorm (f m)) ( …
  -/
  simpa only [← opNNNorm_le_iff] using isLeast_Ici
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias isLeast_op_nnnorm := isLeast_opNNNorm


theorem opNNNorm_prod (f : ContinuousMultilinearMap 𝕜 E G) (g : ContinuousMultilinearMap 𝕜 E G') :
    ‖f.prod g‖₊ = max ‖f‖₊ ‖g‖₊ :=
  eq_of_forall_ge_iff fun _ ↦ by
    /-
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      G' : Type wG'
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁴ : SeminormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      inst✝² : SeminormedAddCommGroup G'
      inst✝¹ : NormedSpace 𝕜 G'
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E G
      g : ContinuousMultilinearMap 𝕜 E G'
      x✝ : NNReal
      ⊢ Iff (LE.le (NNNorm.nnnorm (f.prod g)) x✝) (LE.le (Max.max (NNNorm.nnnorm f)  …
    -/
    simp only [opNNNorm_le_iff, prod_apply, Prod.nnnorm_def', max_le_iff, forall_and]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-02-02")] alias op_nnnorm_prod := opNNNorm_prod


theorem opNorm_prod (f : ContinuousMultilinearMap 𝕜 E G) (g : ContinuousMultilinearMap 𝕜 E G') :
    ‖f.prod g‖ = max ‖f‖ ‖g‖ :=
  congr_arg NNReal.toReal (opNNNorm_prod f g)


@[deprecated (since := "2024-02-02")] alias op_norm_prod := opNorm_prod


theorem opNNNorm_pi
    [∀ i', SeminormedAddCommGroup (E' i')] [∀ i', NormedSpace 𝕜 (E' i')]
    (f : ∀ i', ContinuousMultilinearMap 𝕜 E (E' i')) : ‖pi f‖₊ = ‖f‖₊ :=
                                 /-
                                   𝕜 : Type u
                                   ι : Type v
                                   ι' : Type v'
                                   E : ι → Type wE
                                   E' : ι' → Type wE'
                                   inst✝⁶ : Fintype ι'
                                   inst✝⁵ : NontriviallyNormedField 𝕜
                                   inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                                   inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                                   inst✝² : Fintype ι
                                   inst✝¹ : (i' : ι') → SeminormedAddCommGroup (E' i')
                                   inst✝ : (i' : ι') → NormedSpace 𝕜 (E' i')
                                   f : (i' : ι') → ContinuousMultilinearMap 𝕜 E (E' i')
                                   x✝ : NNReal
                                   ⊢ Iff (LE.le (NNNorm.nnnorm (ContinuousMultilinearMap.pi f)) x✝) (LE.le (NNNor …
                                 -/
  eq_of_forall_ge_iff fun _ ↦ by simpa [opNNNorm_le_iff, pi_nnnorm_le_iff] using forall_swap
                                 /-
                                   🎉 no goals
                                 -/


theorem opNorm_pi {ι' : Type v'} [Fintype ι'] {E' : ι' → Type wE'}
    [∀ i', SeminormedAddCommGroup (E' i')] [∀ i', NormedSpace 𝕜 (E' i')]
    (f : ∀ i', ContinuousMultilinearMap 𝕜 E (E' i')) :
    ‖pi f‖ = ‖f‖ :=
  congr_arg NNReal.toReal (opNNNorm_pi f)


@[deprecated (since := "2024-02-02")] alias op_norm_pi := opNorm_pi


@[simp]
theorem norm_ofSubsingleton [Subsingleton ι] (i : ι) (f : G →L[𝕜] G') :
    ‖ofSubsingleton 𝕜 G G' i f‖ = ‖f‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    G' : Type wG'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : SeminormedAddCommGroup G'
    inst✝² : NormedSpace 𝕜 G'
    inst✝¹ : Fintype ι
    inst✝ : Subsingleton ι
    i : ι
    f : ContinuousLinearMap (RingHom.id 𝕜) G G'
    ⊢ Eq (Norm.norm ((ContinuousMultilinearMap.ofSubsingleton 𝕜 G G' i) f)) (Norm. …
  -/
  letI : Unique ι := uniqueOfSubsingleton i
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    G' : Type wG'
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : SeminormedAddCommGroup G
    inst✝⁴ : NormedSpace 𝕜 G
    inst✝³ : SeminormedAddCommGroup G'
    inst✝² : NormedSpace 𝕜 G'
    inst✝¹ : Fintype ι
    inst✝ : Subsingleton ι
    i : ι
    f : ContinuousLinearMap (RingHom.id 𝕜) G G'
    this : Unique ι := uniqueOfSubsingleton i
    ⊢ Eq (Norm.norm ((ContinuousMultilinearMap.ofSubsingleton 𝕜 G G' i) f)) (Norm. …
  -/
  simp [norm_def, ContinuousLinearMap.norm_def, (Equiv.funUnique _ _).symm.surjective.forall]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_ofSubsingleton [Subsingleton ι] (i : ι) (f : G →L[𝕜] G') :
    ‖ofSubsingleton 𝕜 G G' i f‖₊ = ‖f‖₊ :=
  NNReal.eq <| norm_ofSubsingleton i f


/-- Linear isometry between continuous linear maps from `G` to `G'`
and continuous `1`-multilinear maps from `G` to `G'`. -/
@[simps apply symm_apply]
def ofSubsingletonₗᵢ [Subsingleton ι] (i : ι) :
    (G →L[𝕜] G') ≃ₗᵢ[𝕜] ContinuousMultilinearMap 𝕜 (fun _ : ι ↦ G) G' :=
  { ofSubsingleton 𝕜 G G' i with
    map_add' := fun _ _ ↦ rfl
    map_smul' := fun _ _ ↦ rfl
    norm_map' := norm_ofSubsingleton i }


theorem norm_ofSubsingleton_id_le [Subsingleton ι] (i : ι) :
    ‖ofSubsingleton 𝕜 G G i (.id _ _)‖ ≤ 1 := by
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Fintype ι
    inst✝ : Subsingleton ι
    i : ι
    ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.ofSubsingleton 𝕜 G G i) (Continu …
  -/
  rw [norm_ofSubsingleton]
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Fintype ι
    inst✝ : Subsingleton ι
    i : ι
    ⊢ LE.le (Norm.norm (ContinuousLinearMap.id 𝕜 G)) 1
  -/
  apply ContinuousLinearMap.norm_id_le
  /-
    🎉 no goals
  -/


theorem nnnorm_ofSubsingleton_id_le [Subsingleton ι] (i : ι) :
    ‖ofSubsingleton 𝕜 G G i (.id _ _)‖₊ ≤ 1 :=
  norm_ofSubsingleton_id_le _ _ _


@[simp]
theorem norm_constOfIsEmpty [IsEmpty ι] (x : G) : ‖constOfIsEmpty 𝕜 E x‖ = ‖x‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Fintype ι
    inst✝ : IsEmpty ι
    x : G
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.constOfIsEmpty 𝕜 E x)) (Norm.norm x)
  -/
  apply le_antisymm
    /-
      case a
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : SeminormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : Fintype ι
      inst✝ : IsEmpty ι
      x : G
      ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.constOfIsEmpty 𝕜 E x)) (Norm.norm …
    -/
  · refine opNorm_le_bound (norm_nonneg _) fun x => ?_
    /-
      case a
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : SeminormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : Fintype ι
      inst✝ : IsEmpty ι
      x✝ : G
      x : (i : ι) → E i
      ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.constOfIsEmpty 𝕜 E x✝) x)) (HMul …
    -/
    rw [Fintype.prod_empty, mul_one, constOfIsEmpty_apply]
    /-
      🎉 no goals
    -/
    /-
      case a
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁴ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝³ : SeminormedAddCommGroup G
      inst✝² : NormedSpace 𝕜 G
      inst✝¹ : Fintype ι
      inst✝ : IsEmpty ι
      x : G
      ⊢ LE.le (Norm.norm x) (Norm.norm (ContinuousMultilinearMap.constOfIsEmpty 𝕜 E  …
    -/
  · simpa using (constOfIsEmpty 𝕜 E x).le_opNorm 0
    /-
      🎉 no goals
    -/


@[simp]
theorem nnnorm_constOfIsEmpty [IsEmpty ι] (x : G) : ‖constOfIsEmpty 𝕜 E x‖₊ = ‖x‖₊ :=
  NNReal.eq <| norm_constOfIsEmpty _ _ _


/-- `ContinuousMultilinearMap.prod` as a `LinearIsometryEquiv`. -/
@[simps]
def prodL :
    ContinuousMultilinearMap 𝕜 E G × ContinuousMultilinearMap 𝕜 E G' ≃ₗᵢ[𝕜]
      ContinuousMultilinearMap 𝕜 E (G × G') where
  toFun f := f.1.prod f.2
  invFun f :=
    ((ContinuousLinearMap.fst 𝕜 G G').compContinuousMultilinearMap f,
      (ContinuousLinearMap.snd 𝕜 G G').compContinuousMultilinearMap f)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
                   /-
                     𝕜 : Type u
                     ι : Type v
                     ι' : Type v'
                     E : ι → Type wE
                     E₁ : ι → Type wE₁
                     E' : ι' → Type wE'
                     G : Type wG
                     G' : Type wG'
                     inst✝¹³ : Fintype ι'
                     inst✝¹² : NontriviallyNormedField 𝕜
                     inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
                     inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
                     inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                     inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                     inst✝⁷ : SeminormedAddCommGroup G
                     inst✝⁶ : NormedSpace 𝕜 G
                     inst✝⁵ : SeminormedAddCommGroup G'
                     inst✝⁴ : NormedSpace 𝕜 G'
                     inst✝³ : Fintype ι
                     𝕜' : Type u_1
                     inst✝² : NormedField 𝕜'
                     inst✝¹ : NormedSpace 𝕜' G
                     inst✝ : SMulCommClass 𝕜 𝕜' G
                     f : Prod (ContinuousMultilinearMap 𝕜 E G) (ContinuousMultilinearMap 𝕜 E G')
                     ⊢ Eq ((fun f => { fst := (ContinuousLinearMap.fst 𝕜 G G').compContinuousMultil …
                   -/
                           /-
                             🎉 no goals
                           -/
  left_inv f := by ext <;> rfl
                           /-
                             🎉 no goals
                           -/
                    /-
                      𝕜 : Type u
                      ι : Type v
                      ι' : Type v'
                      E : ι → Type wE
                      E₁ : ι → Type wE₁
                      E' : ι' → Type wE'
                      G : Type wG
                      G' : Type wG'
                      inst✝¹³ : Fintype ι'
                      inst✝¹² : NontriviallyNormedField 𝕜
                      inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
                      inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
                      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                      inst✝⁷ : SeminormedAddCommGroup G
                      inst✝⁶ : NormedSpace 𝕜 G
                      inst✝⁵ : SeminormedAddCommGroup G'
                      inst✝⁴ : NormedSpace 𝕜 G'
                      inst✝³ : Fintype ι
                      𝕜' : Type u_1
                      inst✝² : NormedField 𝕜'
                      inst✝¹ : NormedSpace 𝕜' G
                      inst✝ : SMulCommClass 𝕜 𝕜' G
                      f : ContinuousMultilinearMap 𝕜 E (Prod G G')
                      ⊢ Eq ({ toFun := fun f => f.1.prod f.2, map_add' := ⋯, map_smul' := ⋯ }.toFun  …
                    -/
                            /-
                              🎉 no goals
                            -/
  right_inv f := by ext <;> rfl
                            /-
                              🎉 no goals
                            -/
  norm_map' f := opNorm_prod f.1 f.2


/-- `ContinuousMultilinearMap.pi` as a `LinearIsometryEquiv`. -/
@[simps! apply symm_apply]
def piₗᵢ {ι' : Type v'} [Fintype ι'] {E' : ι' → Type wE'} [∀ i', NormedAddCommGroup (E' i')]
    [∀ i', NormedSpace 𝕜 (E' i')] :
    (Π i', ContinuousMultilinearMap 𝕜 E (E' i'))
      ≃ₗᵢ[𝕜] (ContinuousMultilinearMap 𝕜 E (Π i, E' i)) where
  toLinearEquiv := piLinearEquiv
  norm_map' := opNorm_pi


@[simp]
theorem norm_restrictScalars (f : ContinuousMultilinearMap 𝕜 E G) :
    ‖f.restrictScalars 𝕜'‖ = ‖f‖ :=
  rfl


/-- `ContinuousMultilinearMap.restrictScalars` as a `LinearIsometry`. -/
def restrictScalarsₗᵢ : ContinuousMultilinearMap 𝕜 E G →ₗᵢ[𝕜'] ContinuousMultilinearMap 𝕜' E G where
  toFun := restrictScalars 𝕜'
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  norm_map' _ := rfl


/-- The difference `f m₁ - f m₂` is controlled in terms of `‖f‖` and `‖m₁ - m₂‖`, precise version.
For a less precise but more usable version, see `norm_image_sub_le`. The bound reads
`‖f m - f m'‖ ≤
  ‖f‖ * ‖m 1 - m' 1‖ * max ‖m 2‖ ‖m' 2‖ * max ‖m 3‖ ‖m' 3‖ * ... * max ‖m n‖ ‖m' n‖ + ...`,
where the other terms in the sum are the same products where `1` is replaced by any `i`. -/
theorem norm_image_sub_le' [DecidableEq ι] (f : ContinuousMultilinearMap 𝕜 E G) (m₁ m₂ : ∀ i, E i) :
    ‖f m₁ - f m₂‖ ≤ ‖f‖ * ∑ i, ∏ j, if j = i then ‖m₁ i - m₂ i‖ else max ‖m₁ j‖ ‖m₂ j‖ :=
  f.toMultilinearMap.norm_image_sub_le_of_bound' (norm_nonneg _) f.le_opNorm _ _


/-- The difference `f m₁ - f m₂` is controlled in terms of `‖f‖` and `‖m₁ - m₂‖`, less precise
version. For a more precise but less usable version, see `norm_image_sub_le'`.
The bound is `‖f m - f m'‖ ≤ ‖f‖ * card ι * ‖m - m'‖ * (max ‖m‖ ‖m'‖) ^ (card ι - 1)`. -/
theorem norm_image_sub_le (f : ContinuousMultilinearMap 𝕜 E G) (m₁ m₂ : ∀ i, E i) :
    ‖f m₁ - f m₂‖ ≤ ‖f‖ * Fintype.card ι * max ‖m₁‖ ‖m₂‖ ^ (Fintype.card ι - 1) * ‖m₁ - m₂‖ :=
  f.toMultilinearMap.norm_image_sub_le_of_bound (norm_nonneg _) f.le_opNorm _ _


/-- If a continuous multilinear map is constructed from a multilinear map via the constructor
`mkContinuous`, then its norm is bounded by the bound given to the constructor if it is
nonnegative. -/
theorem MultilinearMap.mkContinuous_norm_le (f : MultilinearMap 𝕜 E G) {C : ℝ} (hC : 0 ≤ C)
    (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) : ‖f.mkContinuous C H‖ ≤ C :=
  ContinuousMultilinearMap.opNorm_le_bound hC fun m => H m


/-- If a continuous multilinear map is constructed from a multilinear map via the constructor
`mkContinuous`, then its norm is bounded by the bound given to the constructor if it is
nonnegative. -/
theorem MultilinearMap.mkContinuous_norm_le' (f : MultilinearMap 𝕜 E G) {C : ℝ}
    (H : ∀ m, ‖f m‖ ≤ C * ∏ i, ‖m i‖) : ‖f.mkContinuous C H‖ ≤ max C 0 :=
  ContinuousMultilinearMap.opNorm_le_bound (le_max_right _ _) fun m ↦ (H m).trans <|
                                                       /-
                                                         𝕜 : Type u
                                                         ι : Type v
                                                         E : ι → Type wE
                                                         G : Type wG
                                                         inst✝⁵ : NontriviallyNormedField 𝕜
                                                         inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
                                                         inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
                                                         inst✝² : SeminormedAddCommGroup G
                                                         inst✝¹ : NormedSpace 𝕜 G
                                                         inst✝ : Fintype ι
                                                         f : MultilinearMap 𝕜 E G
                                                         C : Real
                                                         H : ∀ (m : (i : ι) → E i), LE.le (Norm.norm (f m)) (HMul.hMul C (Finset.univ.p …
                                                         m : (i : ι) → E i
                                                         ⊢ LE.le 0 (Finset.univ.prod fun i => Norm.norm (m i))
                                                       -/
    mul_le_mul_of_nonneg_right (le_max_left _ _) <| by positivity
                                                       /-
                                                         🎉 no goals
                                                       -/


/-- Given a continuous multilinear map `f` on `n` variables (parameterized by `Fin n`) and a subset
`s` of `k` of these variables, one gets a new continuous multilinear map on `Fin k` by varying
these variables, and fixing the other ones equal to a given value `z`. It is denoted by
`f.restr s hk z`, where `hk` is a proof that the cardinality of `s` is `k`. The implicit
identification between `Fin k` and `s` that we use is the canonical (increasing) bijection. -/
def restr {k n : ℕ} (f : (G[×n]→L[𝕜] G' : _)) (s : Finset (Fin n)) (hk : #s = k) (z : G) :
    G[×k]→L[𝕜] G' :=
  (f.toMultilinearMap.restr s hk z).mkContinuous (‖f‖ * ‖z‖ ^ (n - k)) fun _ =>
    MultilinearMap.restr_norm_le _ _ _ _ f.le_opNorm _


theorem norm_restr {k n : ℕ} (f : G[×n]→L[𝕜] G') (s : Finset (Fin n)) (hk : #s = k) (z : G) :
    ‖f.restr s hk z‖ ≤ ‖f‖ * ‖z‖ ^ (n - k) := by
  /-
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : SeminormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    s : Finset (Fin n)
    hk : Eq s.card k
    z : G
    ⊢ LE.le (Norm.norm (f.restr s hk z)) (HMul.hMul (Norm.norm f) (HPow.hPow (Norm …
  -/
  apply MultilinearMap.mkContinuous_norm_le
  /-
    case hC
    𝕜 : Type u
    G : Type wG
    G' : Type wG'
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : SeminormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : SeminormedAddCommGroup G'
    inst✝ : NormedSpace 𝕜 G'
    k n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => G) G'
    s : Finset (Fin n)
    hk : Eq s.card k
    z : G
    ⊢ LE.le 0 (HMul.hMul (Norm.norm f) (HPow.hPow (Norm.norm z) (HSub.hSub n k)))
  -/
  exact mul_nonneg (norm_nonneg _) (pow_nonneg (norm_nonneg _) _)
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_mkPiAlgebra_le [Nonempty ι] : ‖ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A‖ ≤ 1 := by
  /-
    𝕜 : Type u
    ι : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : Fintype ι
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : Nonempty ι
    ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) 1
  -/
  refine opNorm_le_bound zero_le_one fun m => ?_
  /-
    𝕜 : Type u
    ι : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : Fintype ι
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : Nonempty ι
    m : ι → A
    ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A) m)) (HMul.hMu …
  -/
  simp only [ContinuousMultilinearMap.mkPiAlgebra_apply, one_mul]
  /-
    𝕜 : Type u
    ι : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : Fintype ι
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : Nonempty ι
    m : ι → A
    ⊢ LE.le (Norm.norm (Finset.univ.prod fun i => m i)) (Finset.univ.prod fun i => …
  -/
  exact norm_prod_le' _ univ_nonempty _
  /-
    🎉 no goals
  -/


theorem norm_mkPiAlgebra_of_empty [IsEmpty ι] :
    ‖ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A‖ = ‖(1 : A)‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : Fintype ι
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : IsEmpty ι
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) (Norm.norm 1)
  -/
  apply le_antisymm
    /-
      case a
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsEmpty ι
      ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) (Norm.norm 1)
    -/
                              /-
                                🎉 no goals
                              -/
  · apply opNorm_le_bound <;> simp
                              /-
                                🎉 no goals
                              -/
  · -- Porting note: have to annotate types to get mvars to unify
    /-
      case a
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsEmpty ι
      ⊢ LE.le (Norm.norm 1) (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A))
    -/
    convert ratio_le_opNorm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A) fun _ => (1 : A)
    /-
      case h.e'_3
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : IsEmpty ι
      ⊢ Eq (Norm.norm 1) (HDiv.hDiv (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebr …
    -/
    simp [eq_empty_of_isEmpty (univ : Finset ι)]
    /-
      🎉 no goals
    -/


@[simp]
theorem norm_mkPiAlgebra [NormOneClass A] : ‖ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A‖ = 1 := by
  /-
    𝕜 : Type u
    ι : Type v
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : Fintype ι
    A : Type u_1
    inst✝² : NormedCommRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : NormOneClass A
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) 1
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      h✝ : IsEmpty ι
      ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) 1
    -/
  · simp [norm_mkPiAlgebra_of_empty]
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      h✝ : Nonempty ι
      ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A)) 1
    -/
  · refine le_antisymm norm_mkPiAlgebra_le ?_
    /-
      case inr
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      h✝ : Nonempty ι
      ⊢ LE.le 1 (Norm.norm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A))
    -/
    convert ratio_le_opNorm (ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A) fun _ => 1
    /-
      case h.e'_3
      𝕜 : Type u
      ι : Type v
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : Fintype ι
      A : Type u_1
      inst✝² : NormedCommRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      h✝ : Nonempty ι
      ⊢ Eq 1 (HDiv.hDiv (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebra 𝕜 ι A) fun …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem norm_mkPiAlgebraFin_succ_le : ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n.succ A‖ ≤ 1 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n.succ A)) 1
  -/
  refine opNorm_le_bound zero_le_one fun m => ?_
  simp only [ContinuousMultilinearMap.mkPiAlgebraFin_apply, one_mul, List.ofFn_eq_map,
    Fin.prod_univ_def, Multiset.map_coe, Multiset.prod_coe]
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    m : Fin n.succ → A
    ⊢ LE.le (Norm.norm (List.map m (List.finRange n.succ)).prod) (List.map (fun i  …
  -/
  refine (List.norm_prod_le' ?_).trans_eq ?_
    /-
      case refine_1
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      n : Nat
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      m : Fin n.succ → A
      ⊢ Ne (List.map m (List.finRange n.succ)) List.nil
    -/
  · rw [Ne, List.map_eq_nil_iff, List.finRange_eq_nil]
    /-
      case refine_1
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      n : Nat
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      m : Fin n.succ → A
      ⊢ Not (Eq n.succ 0)
    -/
    exact Nat.succ_ne_zero _
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    m : Fin n.succ → A
    ⊢ Eq (List.map Norm.norm (List.map m (List.finRange n.succ))).prod (List.map ( …
  -/
  rw [List.map_map, Function.comp_def]
  /-
    🎉 no goals
  -/


theorem norm_mkPiAlgebraFin_le_of_pos (hn : 0 < n) :
    ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A‖ ≤ 1 := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    hn : LT.lt 0 n
    ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A)) 1
  -/
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hn.ne'
  /-
    case intro
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n.succ A)) 1
  -/
  exact norm_mkPiAlgebraFin_succ_le
  /-
    🎉 no goals
  -/


theorem norm_mkPiAlgebraFin_zero : ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A‖ = ‖(1 : A)‖ := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A)) (Norm.norm 1)
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A)) (Norm.norm …
    -/
  · refine opNorm_le_bound (norm_nonneg (1 : A)) ?_
    /-
      case refine_1
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      ⊢ ∀ (m : Fin 0 → A), LE.le (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebraFi …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      ⊢ LE.le (Norm.norm 1) (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0  …
    -/
  · convert ratio_le_opNorm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A) fun _ => (1 : A)
    /-
      case h.e'_3
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      ⊢ Eq (Norm.norm 1) (HDiv.hDiv (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebr …
    -/
    simp
    /-
      🎉 no goals
    -/


theorem norm_mkPiAlgebraFin_le :
    ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A‖ ≤ max 1 ‖(1 : A)‖ := by
  /-
    𝕜 : Type u
    inst✝² : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝¹ : NormedRing A
    inst✝ : NormedAlgebra 𝕜 A
    ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A)) (Max.max 1 …
  -/
  cases n
    /-
      case zero
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A)) (Max.max 1 …
    -/
  · exact norm_mkPiAlgebraFin_zero.le.trans (le_max_right _ _)
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u
      inst✝² : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝¹ : NormedRing A
      inst✝ : NormedAlgebra 𝕜 A
      n✝ : Nat
      ⊢ LE.le (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 (HAdd.hAdd n✝ 1) …
    -/
  · exact (norm_mkPiAlgebraFin_le_of_pos (Nat.zero_lt_succ _)).trans (le_max_left _ _)
    /-
      🎉 no goals
    -/


@[simp]
theorem norm_mkPiAlgebraFin [NormOneClass A] :
    ‖ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A‖ = 1 := by
  /-
    𝕜 : Type u
    inst✝³ : NontriviallyNormedField 𝕜
    n : Nat
    A : Type u_1
    inst✝² : NormedRing A
    inst✝¹ : NormedAlgebra 𝕜 A
    inst✝ : NormOneClass A
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n A)) 1
  -/
  cases n
    /-
      case zero
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 0 A)) 1
    -/
  · rw [norm_mkPiAlgebraFin_zero]
    /-
      case zero
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      ⊢ Eq (Norm.norm 1) 1
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      n✝ : Nat
      ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 (HAdd.hAdd n✝ 1) A) …
    -/
  · refine le_antisymm norm_mkPiAlgebraFin_succ_le ?_
    refine le_of_eq_of_le ?_ <|
      ratio_le_opNorm (ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 (Nat.succ _) A) fun _ => 1
    /-
      case succ
      𝕜 : Type u
      inst✝³ : NontriviallyNormedField 𝕜
      A : Type u_1
      inst✝² : NormedRing A
      inst✝¹ : NormedAlgebra 𝕜 A
      inst✝ : NormOneClass A
      n✝ : Nat
      ⊢ Eq 1 (HDiv.hDiv (Norm.norm ((ContinuousMultilinearMap.mkPiAlgebraFin 𝕜 n✝.su …
    -/
    simp
    /-
      🎉 no goals
    -/


@[simp]
theorem nnnorm_smulRight (f : ContinuousMultilinearMap 𝕜 E 𝕜) (z : G) :
    ‖f.smulRight z‖₊ = ‖f‖₊ * ‖z‖₊ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    f : ContinuousMultilinearMap 𝕜 E 𝕜
    z : G
    ⊢ Eq (NNNorm.nnnorm (f.smulRight z)) (HMul.hMul (NNNorm.nnnorm f) (NNNorm.nnno …
  -/
  refine le_antisymm ?_ ?_
    /-
      case refine_1
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      ⊢ LE.le (NNNorm.nnnorm (f.smulRight z)) (HMul.hMul (NNNorm.nnnorm f) (NNNorm.n …
    -/
  · refine opNNNorm_le_iff.2 fun m => (nnnorm_smul_le _ _).trans ?_
    /-
      case refine_1
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      m : (i : ι) → E i
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (LinearMap.id (f.toMultilinearMap m))) (NNNo …
    -/
    rw [mul_right_comm]
    /-
      case refine_1
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      m : (i : ι) → E i
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (LinearMap.id (f.toMultilinearMap m))) (NNNo …
    -/
    gcongr
    /-
      case refine_1.bc
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      m : (i : ι) → E i
      ⊢ LE.le (NNNorm.nnnorm (LinearMap.id (f.toMultilinearMap m))) (HMul.hMul (NNNo …
    -/
    exact le_opNNNorm _ _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm f) (NNNorm.nnnorm z)) (NNNorm.nnnorm (f.smul …
    -/
  · obtain hz | hz := eq_zero_or_pos ‖z‖₊
      /-
        case refine_2.inl
        𝕜 : Type u
        ι : Type v
        E : ι → Type wE
        G : Type wG
        inst✝⁵ : NontriviallyNormedField 𝕜
        inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝² : SeminormedAddCommGroup G
        inst✝¹ : NormedSpace 𝕜 G
        inst✝ : Fintype ι
        f : ContinuousMultilinearMap 𝕜 E 𝕜
        z : G
        hz : Eq (NNNorm.nnnorm z) 0
        ⊢ LE.le (HMul.hMul (NNNorm.nnnorm f) (NNNorm.nnnorm z)) (NNNorm.nnnorm (f.smul …
      -/
    · simp [hz]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.inr
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      hz : LT.lt 0 (NNNorm.nnnorm z)
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm f) (NNNorm.nnnorm z)) (NNNorm.nnnorm (f.smul …
    -/
    rw [← le_div_iff₀ hz, opNNNorm_le_iff]
    /-
      case refine_2.inr
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      hz : LT.lt 0 (NNNorm.nnnorm z)
      ⊢ ∀ (m : (i : ι) → E i), LE.le (NNNorm.nnnorm (f m)) (HMul.hMul (HDiv.hDiv (NN …
    -/
    intro m
    /-
      case refine_2.inr
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      hz : LT.lt 0 (NNNorm.nnnorm z)
      m : (i : ι) → E i
      ⊢ LE.le (NNNorm.nnnorm (f m)) (HMul.hMul (HDiv.hDiv (NNNorm.nnnorm (f.smulRigh …
    -/
    rw [div_mul_eq_mul_div, le_div_iff₀ hz]
    /-
      case refine_2.inr
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      hz : LT.lt 0 (NNNorm.nnnorm z)
      m : (i : ι) → E i
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (f m)) (NNNorm.nnnorm z)) (HMul.hMul (NNNorm …
    -/
    refine le_trans ?_ ((f.smulRight z).le_opNNNorm m)
    /-
      case refine_2.inr
      𝕜 : Type u
      ι : Type v
      E : ι → Type wE
      G : Type wG
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝² : SeminormedAddCommGroup G
      inst✝¹ : NormedSpace 𝕜 G
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 E 𝕜
      z : G
      hz : LT.lt 0 (NNNorm.nnnorm z)
      m : (i : ι) → E i
      ⊢ LE.le (HMul.hMul (NNNorm.nnnorm (f m)) (NNNorm.nnnorm z)) (NNNorm.nnnorm ((f …
    -/
    rw [smulRight_apply, nnnorm_smul]
    /-
      🎉 no goals
    -/


@[simp]
theorem norm_smulRight (f : ContinuousMultilinearMap 𝕜 E 𝕜) (z : G) :
    ‖f.smulRight z‖ = ‖f‖ * ‖z‖ :=
  congr_arg NNReal.toReal (nnnorm_smulRight f z)


@[simp]
theorem norm_mkPiRing (z : G) : ‖ContinuousMultilinearMap.mkPiRing 𝕜 ι z‖ = ‖z‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    z : G
    ⊢ Eq (Norm.norm (ContinuousMultilinearMap.mkPiRing 𝕜 ι z)) (Norm.norm z)
  -/
  rw [ContinuousMultilinearMap.mkPiRing, norm_smulRight, norm_mkPiAlgebra, one_mul]
  /-
    🎉 no goals
  -/


variable (𝕜 E G) in
/-- Continuous bilinear map realizing `(f, z) ↦ f.smulRight z`. -/
def smulRightL : ContinuousMultilinearMap 𝕜 E 𝕜 →L[𝕜] G →L[𝕜] ContinuousMultilinearMap 𝕜 E G :=
  LinearMap.mkContinuous₂
    { toFun := fun f ↦
        { toFun := fun z ↦ f.smulRight z
                                   /-
                                     𝕜 : Type u
                                     ι : Type v
                                     ι' : Type v'
                                     E : ι → Type wE
                                     E₁ : ι → Type wE₁
                                     E' : ι' → Type wE'
                                     G : Type wG
                                     G' : Type wG'
                                     inst✝¹⁰ : Fintype ι'
                                     inst✝⁹ : NontriviallyNormedField 𝕜
                                     inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                                     inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                     inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                     inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                     inst✝⁴ : SeminormedAddCommGroup G
                                     inst✝³ : NormedSpace 𝕜 G
                                     inst✝² : SeminormedAddCommGroup G'
                                     inst✝¹ : NormedSpace 𝕜 G'
                                     inst✝ : Fintype ι
                                     f : ContinuousMultilinearMap 𝕜 E 𝕜
                                     x y : G
                                     ⊢ Eq ((fun z => f.smulRight z) (HAdd.hAdd x y)) (HAdd.hAdd ((fun z => f.smulRi …
                                   -/
          map_add' := fun x y ↦ by ext; simp
                                        /-
                                          🎉 no goals
                                        -/
                                    /-
                                      𝕜 : Type u
                                      ι : Type v
                                      ι' : Type v'
                                      E : ι → Type wE
                                      E₁ : ι → Type wE₁
                                      E' : ι' → Type wE'
                                      G : Type wG
                                      G' : Type wG'
                                      inst✝¹⁰ : Fintype ι'
                                      inst✝⁹ : NontriviallyNormedField 𝕜
                                      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                                      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                      inst✝⁴ : SeminormedAddCommGroup G
                                      inst✝³ : NormedSpace 𝕜 G
                                      inst✝² : SeminormedAddCommGroup G'
                                      inst✝¹ : NormedSpace 𝕜 G'
                                      inst✝ : Fintype ι
                                      f : ContinuousMultilinearMap 𝕜 E 𝕜
                                      c : 𝕜
                                      x : G
                                      ⊢ Eq ({ toFun := fun z => f.smulRight z, map_add' := ⋯ }.toFun (HSMul.hSMul c  …
                                    -/
          map_smul' := fun c x ↦ by ext; simp [smul_smul, mul_comm] }
                                         /-
                                           🎉 no goals
                                         -/
                               /-
                                 𝕜 : Type u
                                 ι : Type v
                                 ι' : Type v'
                                 E : ι → Type wE
                                 E₁ : ι → Type wE₁
                                 E' : ι' → Type wE'
                                 G : Type wG
                                 G' : Type wG'
                                 inst✝¹⁰ : Fintype ι'
                                 inst✝⁹ : NontriviallyNormedField 𝕜
                                 inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                                 inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                 inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                 inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                 inst✝⁴ : SeminormedAddCommGroup G
                                 inst✝³ : NormedSpace 𝕜 G
                                 inst✝² : SeminormedAddCommGroup G'
                                 inst✝¹ : NormedSpace 𝕜 G'
                                 inst✝ : Fintype ι
                                 f g : ContinuousMultilinearMap 𝕜 E 𝕜
                                 ⊢ Eq ((fun f => { toFun := fun z => f.smulRight z, map_add' := ⋯, map_smul' := …
                               -/
      map_add' := fun f g ↦ by ext; simp [add_smul]
                                    /-
                                      🎉 no goals
                                    -/
                                /-
                                  𝕜 : Type u
                                  ι : Type v
                                  ι' : Type v'
                                  E : ι → Type wE
                                  E₁ : ι → Type wE₁
                                  E' : ι' → Type wE'
                                  G : Type wG
                                  G' : Type wG'
                                  inst✝¹⁰ : Fintype ι'
                                  inst✝⁹ : NontriviallyNormedField 𝕜
                                  inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                                  inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                  inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                  inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                  inst✝⁴ : SeminormedAddCommGroup G
                                  inst✝³ : NormedSpace 𝕜 G
                                  inst✝² : SeminormedAddCommGroup G'
                                  inst✝¹ : NormedSpace 𝕜 G'
                                  inst✝ : Fintype ι
                                  c : 𝕜
                                  f : ContinuousMultilinearMap 𝕜 E 𝕜
                                  ⊢ Eq ({ toFun := fun f => { toFun := fun z => f.smulRight z, map_add' := ⋯, ma …
                                -/
      map_smul' := fun c f ↦ by ext; simp [smul_smul] }
                                     /-
                                       🎉 no goals
                                     -/
                    /-
                      𝕜 : Type u
                      ι : Type v
                      ι' : Type v'
                      E : ι → Type wE
                      E₁ : ι → Type wE₁
                      E' : ι' → Type wE'
                      G : Type wG
                      G' : Type wG'
                      inst✝¹⁰ : Fintype ι'
                      inst✝⁹ : NontriviallyNormedField 𝕜
                      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                      inst✝⁴ : SeminormedAddCommGroup G
                      inst✝³ : NormedSpace 𝕜 G
                      inst✝² : SeminormedAddCommGroup G'
                      inst✝¹ : NormedSpace 𝕜 G'
                      inst✝ : Fintype ι
                      f : ContinuousMultilinearMap 𝕜 E 𝕜
                      z : G
                      ⊢ LE.le (Norm.norm (({ toFun := fun f => { toFun := fun z => f.smulRight z, ma …
                    -/
    1 (fun f z ↦ by simp [norm_smulRight])
                    /-
                      🎉 no goals
                    -/


@[simp] lemma smulRightL_apply (f : ContinuousMultilinearMap 𝕜 E 𝕜) (z : G) :
  smulRightL 𝕜 E G f z = f.smulRight z := rfl


set_option maxSynthPendingDepth 2 in
lemma norm_smulRightL_le : ‖smulRightL 𝕜 E G‖ ≤ 1 :=
  LinearMap.mkContinuous₂_norm_le _ zero_le_one _


/-- Continuous multilinear maps on `𝕜^n` with values in `G` are in bijection with `G`, as such a
continuous multilinear map is completely determined by its value on the constant vector made of
ones. We register this bijection as a linear isometry in
`ContinuousMultilinearMap.piFieldEquiv`. -/
protected def piFieldEquiv : G ≃ₗᵢ[𝕜] ContinuousMultilinearMap 𝕜 (fun _ : ι => 𝕜) G where
  toFun z := ContinuousMultilinearMap.mkPiRing 𝕜 ι z
  invFun f := f fun _ => 1
  map_add' z z' := by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁴ : SeminormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      inst✝² : SeminormedAddCommGroup G'
      inst✝¹ : NormedSpace 𝕜 G'
      inst✝ : Fintype ι
      z z' : G
      ⊢ Eq ((fun z => ContinuousMultilinearMap.mkPiRing 𝕜 ι z) (HAdd.hAdd z z')) (HA …
    -/
    ext m
    /-
      case H
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁴ : SeminormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      inst✝² : SeminormedAddCommGroup G'
      inst✝¹ : NormedSpace 𝕜 G'
      inst✝ : Fintype ι
      z z' : G
      m : ι → 𝕜
      ⊢ Eq (((fun z => ContinuousMultilinearMap.mkPiRing 𝕜 ι z) (HAdd.hAdd z z')) m) …
    -/
    simp [smul_add]
    /-
      🎉 no goals
    -/
  map_smul' c z := by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁴ : SeminormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      inst✝² : SeminormedAddCommGroup G'
      inst✝¹ : NormedSpace 𝕜 G'
      inst✝ : Fintype ι
      c : 𝕜
      z : G
      ⊢ Eq ({ toFun := fun z => ContinuousMultilinearMap.mkPiRing 𝕜 ι z, map_add' := …
    -/
    ext m
    /-
      case H
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹⁰ : Fintype ι'
      inst✝⁹ : NontriviallyNormedField 𝕜
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁴ : SeminormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      inst✝² : SeminormedAddCommGroup G'
      inst✝¹ : NormedSpace 𝕜 G'
      inst✝ : Fintype ι
      c : 𝕜
      z : G
      m : ι → 𝕜
      ⊢ Eq (({ toFun := fun z => ContinuousMultilinearMap.mkPiRing 𝕜 ι z, map_add' : …
    -/
    simp [smul_smul, mul_comm]
    /-
      🎉 no goals
    -/
                   /-
                     𝕜 : Type u
                     ι : Type v
                     ι' : Type v'
                     E : ι → Type wE
                     E₁ : ι → Type wE₁
                     E' : ι' → Type wE'
                     G : Type wG
                     G' : Type wG'
                     inst✝¹⁰ : Fintype ι'
                     inst✝⁹ : NontriviallyNormedField 𝕜
                     inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                     inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                     inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                     inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                     inst✝⁴ : SeminormedAddCommGroup G
                     inst✝³ : NormedSpace 𝕜 G
                     inst✝² : SeminormedAddCommGroup G'
                     inst✝¹ : NormedSpace 𝕜 G'
                     inst✝ : Fintype ι
                     z : G
                     ⊢ Eq ((fun f => f fun x => 1) ({ toFun := fun z => ContinuousMultilinearMap.mk …
                   -/
  left_inv z := by simp
                   /-
                     🎉 no goals
                   -/
  right_inv f := f.mkPiRing_apply_one_eq_self
  norm_map' := norm_mkPiRing


theorem norm_compContinuousMultilinearMap_le (g : G →L[𝕜] G') (f : ContinuousMultilinearMap 𝕜 E G) :
    ‖g.compContinuousMultilinearMap f‖ ≤ ‖g‖ * ‖f‖ :=
                                               /-
                                                 𝕜 : Type u
                                                 ι : Type v
                                                 E : ι → Type wE
                                                 G : Type wG
                                                 G' : Type wG'
                                                 inst✝⁷ : NontriviallyNormedField 𝕜
                                                 inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
                                                 inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
                                                 inst✝⁴ : SeminormedAddCommGroup G
                                                 inst✝³ : NormedSpace 𝕜 G
                                                 inst✝² : SeminormedAddCommGroup G'
                                                 inst✝¹ : NormedSpace 𝕜 G'
                                                 inst✝ : Fintype ι
                                                 g : ContinuousLinearMap (RingHom.id 𝕜) G G'
                                                 f : ContinuousMultilinearMap 𝕜 E G
                                                 ⊢ LE.le 0 (HMul.hMul (Norm.norm g) (Norm.norm f))
                                               -/
  ContinuousMultilinearMap.opNorm_le_bound (by positivity) fun m ↦
                                               /-
                                                 🎉 no goals
                                               -/
    calc
      ‖g (f m)‖ ≤ ‖g‖ * (‖f‖ * ∏ i, ‖m i‖) := g.le_opNorm_of_le <| f.le_opNorm _
      _ = _ := (mul_assoc _ _ _).symm


/-- `ContinuousLinearMap.compContinuousMultilinearMap` as a bundled continuous bilinear map. -/
def compContinuousMultilinearMapL :
    (G →L[𝕜] G') →L[𝕜] ContinuousMultilinearMap 𝕜 E G →L[𝕜] ContinuousMultilinearMap 𝕜 E G' :=
  LinearMap.mkContinuous₂
    (LinearMap.mk₂ 𝕜 compContinuousMultilinearMap (fun _ _ _ => rfl) (fun _ _ _ => rfl)
                         /-
                           𝕜 : Type u
                           ι : Type v
                           ι' : Type v'
                           E : ι → Type wE
                           E₁ : ι → Type wE₁
                           E' : ι' → Type wE'
                           G : Type wG
                           G' : Type wG'
                           inst✝¹⁰ : Fintype ι'
                           inst✝⁹ : NontriviallyNormedField 𝕜
                           inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                           inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                           inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                           inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                           inst✝⁴ : SeminormedAddCommGroup G
                           inst✝³ : NormedSpace 𝕜 G
                           inst✝² : SeminormedAddCommGroup G'
                           inst✝¹ : NormedSpace 𝕜 G'
                           inst✝ : Fintype ι
                           f : ContinuousLinearMap (RingHom.id 𝕜) G G'
                           g₁ g₂ : ContinuousMultilinearMap 𝕜 E G
                           ⊢ Eq (f.compContinuousMultilinearMap (HAdd.hAdd g₁ g₂)) (HAdd.hAdd (f.compCont …
                         -/
      (fun f g₁ g₂ => by ext1; apply f.map_add)
                               /-
                                 🎉 no goals
                               -/
                       /-
                         𝕜 : Type u
                         ι : Type v
                         ι' : Type v'
                         E : ι → Type wE
                         E₁ : ι → Type wE₁
                         E' : ι' → Type wE'
                         G : Type wG
                         G' : Type wG'
                         inst✝¹⁰ : Fintype ι'
                         inst✝⁹ : NontriviallyNormedField 𝕜
                         inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                         inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                         inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                         inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                         inst✝⁴ : SeminormedAddCommGroup G
                         inst✝³ : NormedSpace 𝕜 G
                         inst✝² : SeminormedAddCommGroup G'
                         inst✝¹ : NormedSpace 𝕜 G'
                         inst✝ : Fintype ι
                         c : 𝕜
                         f : ContinuousLinearMap (RingHom.id 𝕜) G G'
                         g : ContinuousMultilinearMap 𝕜 E G
                         ⊢ Eq (f.compContinuousMultilinearMap (HSMul.hSMul c g)) (HSMul.hSMul c (f.comp …
                       -/
      (fun c f g => by ext1; simp))
                             /-
                               🎉 no goals
                             -/
    1
                  /-
                    𝕜 : Type u
                    ι : Type v
                    ι' : Type v'
                    E : ι → Type wE
                    E₁ : ι → Type wE₁
                    E' : ι' → Type wE'
                    G : Type wG
                    G' : Type wG'
                    inst✝¹⁰ : Fintype ι'
                    inst✝⁹ : NontriviallyNormedField 𝕜
                    inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                    inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                    inst✝⁴ : SeminormedAddCommGroup G
                    inst✝³ : NormedSpace 𝕜 G
                    inst✝² : SeminormedAddCommGroup G'
                    inst✝¹ : NormedSpace 𝕜 G'
                    inst✝ : Fintype ι
                    f : ContinuousLinearMap (RingHom.id 𝕜) G G'
                    g : ContinuousMultilinearMap 𝕜 E G
                    ⊢ LE.le (Norm.norm (((LinearMap.mk₂ 𝕜 ContinuousLinearMap.compContinuousMultil …
                  -/
    fun f g => by rw [one_mul]; exact f.norm_compContinuousMultilinearMap_le g
                                /-
                                  🎉 no goals
                                -/


/-- `ContinuousLinearMap.compContinuousMultilinearMap` as a bundled
continuous linear equiv. -/
nonrec
def _root_.ContinuousLinearEquiv.compContinuousMultilinearMapL (g : G ≃L[𝕜] G') :
    ContinuousMultilinearMap 𝕜 E G ≃L[𝕜] ContinuousMultilinearMap 𝕜 E G' :=
  { compContinuousMultilinearMapL 𝕜 E G G' g.toContinuousLinearMap with
    invFun := compContinuousMultilinearMapL 𝕜 E G' G g.symm.toContinuousLinearMap
    left_inv := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        ⊢ Function.LeftInverse (⇑((ContinuousLinearMap.compContinuousMultilinearMapL 𝕜 …
      -/
      intro f
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        f : ContinuousMultilinearMap 𝕜 E G
        ⊢ Eq (((ContinuousLinearMap.compContinuousMultilinearMapL 𝕜 E G' G) ↑g.symm) ( …
      -/
      ext1 m
      /-
        case H
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        f : ContinuousMultilinearMap 𝕜 E G
        m : (i : ι) → E i
        ⊢ Eq ((((ContinuousLinearMap.compContinuousMultilinearMapL 𝕜 E G' G) ↑g.symm)  …
      -/
      simp [compContinuousMultilinearMapL]
      /-
        🎉 no goals
      -/
    right_inv := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        ⊢ Function.RightInverse (⇑((ContinuousLinearMap.compContinuousMultilinearMapL  …
      -/
      intro f
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        f : ContinuousMultilinearMap 𝕜 E G'
        ⊢ Eq ((↑__src✝).toFun (((ContinuousLinearMap.compContinuousMultilinearMapL 𝕜 E …
      -/
      ext1 m
      /-
        case H
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        g : ContinuousLinearEquiv (RingHom.id 𝕜) G G'
        f : ContinuousMultilinearMap 𝕜 E G'
        m : (i : ι) → E i
        ⊢ Eq (((↑__src✝).toFun (((ContinuousLinearMap.compContinuousMultilinearMapL 𝕜  …
      -/
      simp [compContinuousMultilinearMapL]
      /-
        🎉 no goals
      -/
    continuous_toFun := (compContinuousMultilinearMapL 𝕜 E G G' g.toContinuousLinearMap).continuous
    continuous_invFun :=
      (compContinuousMultilinearMapL 𝕜 E G' G g.symm.toContinuousLinearMap).continuous }


@[simp]
theorem _root_.ContinuousLinearEquiv.compContinuousMultilinearMapL_symm (g : G ≃L[𝕜] G') :
    (g.compContinuousMultilinearMapL E).symm = g.symm.compContinuousMultilinearMapL E :=
  rfl


@[simp]
theorem _root_.ContinuousLinearEquiv.compContinuousMultilinearMapL_apply (g : G ≃L[𝕜] G')
    (f : ContinuousMultilinearMap 𝕜 E G) :
    g.compContinuousMultilinearMapL E f = (g : G →L[𝕜] G').compContinuousMultilinearMap f :=
  rfl


/-- Flip arguments in `f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E G'` to get
`ContinuousMultilinearMap 𝕜 E (G →L[𝕜] G')` -/
@[simps! apply_apply]
def flipMultilinear (f : G →L[𝕜] ContinuousMultilinearMap 𝕜 E G') :
    ContinuousMultilinearMap 𝕜 E (G →L[𝕜] G') :=
  MultilinearMap.mkContinuous
    { toFun := fun m =>
        LinearMap.mkContinuous
          { toFun := fun x => f x m
                                      /-
                                        𝕜 : Type u
                                        ι : Type v
                                        ι' : Type v'
                                        E : ι → Type wE
                                        E₁ : ι → Type wE₁
                                        E' : ι' → Type wE'
                                        G : Type wG
                                        G' : Type wG'
                                        inst✝¹⁰ : Fintype ι'
                                        inst✝⁹ : NontriviallyNormedField 𝕜
                                        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                                        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                                        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                        inst✝⁴ : SeminormedAddCommGroup G
                                        inst✝³ : NormedSpace 𝕜 G
                                        inst✝² : SeminormedAddCommGroup G'
                                        inst✝¹ : NormedSpace 𝕜 G'
                                        inst✝ : Fintype ι
                                        f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
                                        m : (i : ι) → E i
                                        x y : G
                                        ⊢ Eq ((fun x => (f x) m) (HAdd.hAdd x y)) (HAdd.hAdd ((fun x => (f x) m) x) (( …
                                      -/
            map_add' := fun x y => by simp only [map_add, ContinuousMultilinearMap.add_apply]
                                      /-
                                        🎉 no goals
                                      -/
            map_smul' := fun c x => by
              /-
                𝕜 : Type u
                ι : Type v
                ι' : Type v'
                E : ι → Type wE
                E₁ : ι → Type wE₁
                E' : ι' → Type wE'
                G : Type wG
                G' : Type wG'
                inst✝¹⁰ : Fintype ι'
                inst✝⁹ : NontriviallyNormedField 𝕜
                inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                inst✝⁴ : SeminormedAddCommGroup G
                inst✝³ : NormedSpace 𝕜 G
                inst✝² : SeminormedAddCommGroup G'
                inst✝¹ : NormedSpace 𝕜 G'
                inst✝ : Fintype ι
                f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
                m : (i : ι) → E i
                c : 𝕜
                x : G
                ⊢ Eq ({ toFun := fun x => (f x) m, map_add' := ⋯ }.toFun (HSMul.hSMul c x)) (H …
              -/
              simp only [ContinuousMultilinearMap.smul_apply, map_smul, RingHom.id_apply] }
              /-
                🎉 no goals
              -/
          (‖f‖ * ∏ i, ‖m i‖) fun x => by
          /-
            𝕜 : Type u
            ι : Type v
            ι' : Type v'
            E : ι → Type wE
            E₁ : ι → Type wE₁
            E' : ι' → Type wE'
            G : Type wG
            G' : Type wG'
            inst✝¹⁰ : Fintype ι'
            inst✝⁹ : NontriviallyNormedField 𝕜
            inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
            inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
            inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
            inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
            inst✝⁴ : SeminormedAddCommGroup G
            inst✝³ : NormedSpace 𝕜 G
            inst✝² : SeminormedAddCommGroup G'
            inst✝¹ : NormedSpace 𝕜 G'
            inst✝ : Fintype ι
            f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
            m : (i : ι) → E i
            x : G
            ⊢ LE.le (Norm.norm ({ toFun := fun x => (f x) m, map_add' := ⋯, map_smul' := ⋯ …
          -/
          rw [mul_right_comm]
          /-
            𝕜 : Type u
            ι : Type v
            ι' : Type v'
            E : ι → Type wE
            E₁ : ι → Type wE₁
            E' : ι' → Type wE'
            G : Type wG
            G' : Type wG'
            inst✝¹⁰ : Fintype ι'
            inst✝⁹ : NontriviallyNormedField 𝕜
            inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
            inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
            inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
            inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
            inst✝⁴ : SeminormedAddCommGroup G
            inst✝³ : NormedSpace 𝕜 G
            inst✝² : SeminormedAddCommGroup G'
            inst✝¹ : NormedSpace 𝕜 G'
            inst✝ : Fintype ι
            f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
            m : (i : ι) → E i
            x : G
            ⊢ LE.le (Norm.norm ({ toFun := fun x => (f x) m, map_add' := ⋯, map_smul' := ⋯ …
          -/
          exact (f x).le_of_opNorm_le (f.le_opNorm x) _
          /-
            🎉 no goals
          -/
      map_update_add' := fun m i x y => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι'
          inst✝¹⁰ : NontriviallyNormedField 𝕜
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁵ : SeminormedAddCommGroup G
          inst✝⁴ : NormedSpace 𝕜 G
          inst✝³ : SeminormedAddCommGroup G'
          inst✝² : NormedSpace 𝕜 G'
          inst✝¹ : Fintype ι
          f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          x y : E i
          ⊢ Eq ((fun m => { toFun := fun x => (f x) m, map_add' := ⋯, map_smul' := ⋯ }.m …
        -/
        ext1
        simp only [add_apply, ContinuousMultilinearMap.map_update_add, LinearMap.coe_mk,
          LinearMap.mkContinuous_apply, AddHom.coe_mk]
      map_update_smul' := fun m i c x => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹¹ : Fintype ι'
          inst✝¹⁰ : NontriviallyNormedField 𝕜
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁵ : SeminormedAddCommGroup G
          inst✝⁴ : NormedSpace 𝕜 G
          inst✝³ : SeminormedAddCommGroup G'
          inst✝² : NormedSpace 𝕜 G'
          inst✝¹ : Fintype ι
          f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          c : 𝕜
          x : E i
          ⊢ Eq ((fun m => { toFun := fun x => (f x) m, map_add' := ⋯, map_smul' := ⋯ }.m …
        -/
        ext1
        simp only [coe_smul', ContinuousMultilinearMap.map_update_smul, LinearMap.coe_mk,
          LinearMap.mkContinuous_apply, Pi.smul_apply, AddHom.coe_mk] }
    ‖f‖ fun m => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
        m : (i : ι) → E i
        ⊢ LE.le (Norm.norm ({ toFun := fun m => { toFun := fun x => (f x) m, map_add'  …
      -/
      dsimp only [MultilinearMap.coe_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 E G')
        m : (i : ι) → E i
        ⊢ LE.le (Norm.norm ({ toFun := fun x => (f x) m, map_add' := ⋯, map_smul' := ⋯ …
      -/
      exact LinearMap.mkContinuous_norm_le _ (by positivity) _
      /-
        🎉 no goals
      -/


theorem LinearIsometry.norm_compContinuousMultilinearMap (g : G →ₗᵢ[𝕜] G')
    (f : ContinuousMultilinearMap 𝕜 E G) :
    ‖g.toContinuousLinearMap.compContinuousMultilinearMap f‖ = ‖f‖ := by
  simp only [ContinuousLinearMap.compContinuousMultilinearMap_coe,
    LinearIsometry.coe_toContinuousLinearMap, LinearIsometry.norm_map,
    ContinuousMultilinearMap.norm_def, Function.comp_apply]


/-- Given a map `f : G →ₗ[𝕜] MultilinearMap 𝕜 E G'` and an estimate
`H : ∀ x m, ‖f x m‖ ≤ C * ‖x‖ * ∏ i, ‖m i‖`, construct a continuous linear
map from `G` to `ContinuousMultilinearMap 𝕜 E G'`.

In order to lift, e.g., a map `f : (MultilinearMap 𝕜 E G) →ₗ[𝕜] MultilinearMap 𝕜 E' G'`
to a map `(ContinuousMultilinearMap 𝕜 E G) →L[𝕜] ContinuousMultilinearMap 𝕜 E' G'`,
one can apply this construction to `f.comp ContinuousMultilinearMap.toMultilinearMapLinear`
which is a linear map from `ContinuousMultilinearMap 𝕜 E G` to `MultilinearMap 𝕜 E' G'`. -/
def mkContinuousLinear (f : G →ₗ[𝕜] MultilinearMap 𝕜 E G') (C : ℝ)
    (H : ∀ x m, ‖f x m‖ ≤ C * ‖x‖ * ∏ i, ‖m i‖) : G →L[𝕜] ContinuousMultilinearMap 𝕜 E G' :=
  LinearMap.mkContinuous
    { toFun := fun x => (f x).mkContinuous (C * ‖x‖) <| H x
      map_add' := fun x y => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          x y : G
          ⊢ Eq ((fun x => (f x).mkContinuous (HMul.hMul C (Norm.norm x)) ⋯) (HAdd.hAdd x …
        -/
        ext1
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          x y : G
          x✝ : (i : ι) → E i
          ⊢ Eq (((fun x => (f x).mkContinuous (HMul.hMul C (Norm.norm x)) ⋯) (HAdd.hAdd  …
        -/
        simp only [_root_.map_add]
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          x y : G
          x✝ : (i : ι) → E i
          ⊢ Eq (((HAdd.hAdd (f x) (f y)).mkContinuous (HMul.hMul C (Norm.norm (HAdd.hAdd …
        -/
        rfl
        /-
          🎉 no goals
        -/
      map_smul' := fun c x => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          c : 𝕜
          x : G
          ⊢ Eq ({ toFun := fun x => (f x).mkContinuous (HMul.hMul C (Norm.norm x)) ⋯, ma …
        -/
        ext1
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          c : 𝕜
          x : G
          x✝ : (i : ι) → E i
          ⊢ Eq (({ toFun := fun x => (f x).mkContinuous (HMul.hMul C (Norm.norm x)) ⋯, m …
        -/
        simp only [_root_.map_smul]
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
          C : Real
          H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
          c : 𝕜
          x : G
          x✝ : (i : ι) → E i
          ⊢ Eq (((HSMul.hSMul c (f x)).mkContinuous (HMul.hMul C (Norm.norm (HSMul.hSMul …
        -/
        rfl }
        /-
          🎉 no goals
        -/
    (max C 0) fun x => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
        C : Real
        H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
        x : G
        ⊢ LE.le (Norm.norm ({ toFun := fun x => (f x).mkContinuous (HMul.hMul C (Norm. …
      -/
      rw [LinearMap.coe_mk, AddHom.coe_mk] -- Porting note: added
      exact ((f x).mkContinuous_norm_le' _).trans_eq <| by
        rw [max_mul_of_nonneg _ _ (norm_nonneg x), zero_mul]


theorem mkContinuousLinear_norm_le' (f : G →ₗ[𝕜] MultilinearMap 𝕜 E G') (C : ℝ)
    (H : ∀ x m, ‖f x m‖ ≤ C * ‖x‖ * ∏ i, ‖m i‖) : ‖mkContinuousLinear f C H‖ ≤ max C 0 := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    G' : Type wG'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : SeminormedAddCommGroup G'
    inst✝¹ : NormedSpace 𝕜 G'
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
    C : Real
    H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
    ⊢ LE.le (Norm.norm (MultilinearMap.mkContinuousLinear f C H)) (Max.max C 0)
  -/
  dsimp only [mkContinuousLinear]
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    G' : Type wG'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : SeminormedAddCommGroup G'
    inst✝¹ : NormedSpace 𝕜 G'
    inst✝ : Fintype ι
    f : LinearMap (RingHom.id 𝕜) G (MultilinearMap 𝕜 E G')
    C : Real
    H : ∀ (x : G) (m : (i : ι) → E i), LE.le (Norm.norm ((f x) m)) (HMul.hMul (HMu …
    ⊢ LE.le (Norm.norm ({ toFun := fun x => (f x).mkContinuous (HMul.hMul C (Norm. …
  -/
  exact LinearMap.mkContinuous_norm_le _ (le_max_right _ _) _
  /-
    🎉 no goals
  -/


theorem mkContinuousLinear_norm_le (f : G →ₗ[𝕜] MultilinearMap 𝕜 E G') {C : ℝ} (hC : 0 ≤ C)
    (H : ∀ x m, ‖f x m‖ ≤ C * ‖x‖ * ∏ i, ‖m i‖) : ‖mkContinuousLinear f C H‖ ≤ C :=
  (mkContinuousLinear_norm_le' f C H).trans_eq (max_eq_left hC)


/-- Given a map `f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)` and an estimate
`H : ∀ m m', ‖f m m'‖ ≤ C * ∏ i, ‖m i‖ * ∏ i, ‖m' i‖`, upgrade all `MultilinearMap`s in the type to
`ContinuousMultilinearMap`s. -/
def mkContinuousMultilinear (f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)) (C : ℝ)
    (H : ∀ m₁ m₂, ‖f m₁ m₂‖ ≤ (C * ∏ i, ‖m₁ i‖) * ∏ i, ‖m₂ i‖) :
    ContinuousMultilinearMap 𝕜 E (ContinuousMultilinearMap 𝕜 E' G) :=
  mkContinuous
    { toFun := fun m => mkContinuous (f m) (C * ∏ i, ‖m i‖) <| H m
      map_update_add' := fun m i x y => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹³ : Fintype ι'
          inst✝¹² : NontriviallyNormedField 𝕜
          inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁷ : SeminormedAddCommGroup G
          inst✝⁶ : NormedSpace 𝕜 G
          inst✝⁵ : SeminormedAddCommGroup G'
          inst✝⁴ : NormedSpace 𝕜 G'
          inst✝³ : Fintype ι
          inst✝² : (i : ι') → SeminormedAddCommGroup (E' i)
          inst✝¹ : (i : ι') → NormedSpace 𝕜 (E' i)
          f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
          C : Real
          H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          x y : E i
          ⊢ Eq ((fun m => (f m).mkContinuous (HMul.hMul C (Finset.univ.prod fun i => Nor …
        -/
        ext1
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹³ : Fintype ι'
          inst✝¹² : NontriviallyNormedField 𝕜
          inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁷ : SeminormedAddCommGroup G
          inst✝⁶ : NormedSpace 𝕜 G
          inst✝⁵ : SeminormedAddCommGroup G'
          inst✝⁴ : NormedSpace 𝕜 G'
          inst✝³ : Fintype ι
          inst✝² : (i : ι') → SeminormedAddCommGroup (E' i)
          inst✝¹ : (i : ι') → NormedSpace 𝕜 (E' i)
          f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
          C : Real
          H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          x y : E i
          x✝ : (i : ι') → E' i
          ⊢ Eq (((fun m => (f m).mkContinuous (HMul.hMul C (Finset.univ.prod fun i => No …
        -/
        simp
        /-
          🎉 no goals
        -/
      map_update_smul' := fun m i c x => by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹³ : Fintype ι'
          inst✝¹² : NontriviallyNormedField 𝕜
          inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁷ : SeminormedAddCommGroup G
          inst✝⁶ : NormedSpace 𝕜 G
          inst✝⁵ : SeminormedAddCommGroup G'
          inst✝⁴ : NormedSpace 𝕜 G'
          inst✝³ : Fintype ι
          inst✝² : (i : ι') → SeminormedAddCommGroup (E' i)
          inst✝¹ : (i : ι') → NormedSpace 𝕜 (E' i)
          f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
          C : Real
          H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          c : 𝕜
          x : E i
          ⊢ Eq ((fun m => (f m).mkContinuous (HMul.hMul C (Finset.univ.prod fun i => Nor …
        -/
        ext1
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹³ : Fintype ι'
          inst✝¹² : NontriviallyNormedField 𝕜
          inst✝¹¹ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝¹⁰ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁷ : SeminormedAddCommGroup G
          inst✝⁶ : NormedSpace 𝕜 G
          inst✝⁵ : SeminormedAddCommGroup G'
          inst✝⁴ : NormedSpace 𝕜 G'
          inst✝³ : Fintype ι
          inst✝² : (i : ι') → SeminormedAddCommGroup (E' i)
          inst✝¹ : (i : ι') → NormedSpace 𝕜 (E' i)
          f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
          C : Real
          H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
          inst✝ : DecidableEq ι
          m : (i : ι) → E i
          i : ι
          c : 𝕜
          x : E i
          x✝ : (i : ι') → E' i
          ⊢ Eq (((fun m => (f m).mkContinuous (HMul.hMul C (Finset.univ.prod fun i => No …
        -/
        simp }
        /-
          🎉 no goals
        -/
    (max C 0) fun m => by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹² : Fintype ι'
        inst✝¹¹ : NontriviallyNormedField 𝕜
        inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁶ : SeminormedAddCommGroup G
        inst✝⁵ : NormedSpace 𝕜 G
        inst✝⁴ : SeminormedAddCommGroup G'
        inst✝³ : NormedSpace 𝕜 G'
        inst✝² : Fintype ι
        inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
        inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
        f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
        C : Real
        H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
        m : (i : ι) → E i
        ⊢ LE.le (Norm.norm ({ toFun := fun m => (f m).mkContinuous (HMul.hMul C (Finse …
      -/
      simp only [coe_mk]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹² : Fintype ι'
        inst✝¹¹ : NontriviallyNormedField 𝕜
        inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁶ : SeminormedAddCommGroup G
        inst✝⁵ : NormedSpace 𝕜 G
        inst✝⁴ : SeminormedAddCommGroup G'
        inst✝³ : NormedSpace 𝕜 G'
        inst✝² : Fintype ι
        inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
        inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
        f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
        C : Real
        H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
        m : (i : ι) → E i
        ⊢ LE.le (Norm.norm ((f m).mkContinuous (HMul.hMul C (Finset.univ.prod fun i => …
      -/
      refine ((f m).mkContinuous_norm_le' _).trans_eq ?_
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹² : Fintype ι'
        inst✝¹¹ : NontriviallyNormedField 𝕜
        inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁶ : SeminormedAddCommGroup G
        inst✝⁵ : NormedSpace 𝕜 G
        inst✝⁴ : SeminormedAddCommGroup G'
        inst✝³ : NormedSpace 𝕜 G'
        inst✝² : Fintype ι
        inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
        inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
        f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
        C : Real
        H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
        m : (i : ι) → E i
        ⊢ Eq (Max.max (HMul.hMul C (Finset.univ.prod fun i => Norm.norm (m i))) 0) (HM …
      -/
      rw [max_mul_of_nonneg, zero_mul]
      /-
        case hc
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹² : Fintype ι'
        inst✝¹¹ : NontriviallyNormedField 𝕜
        inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁶ : SeminormedAddCommGroup G
        inst✝⁵ : NormedSpace 𝕜 G
        inst✝⁴ : SeminormedAddCommGroup G'
        inst✝³ : NormedSpace 𝕜 G'
        inst✝² : Fintype ι
        inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
        inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
        f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
        C : Real
        H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
        m : (i : ι) → E i
        ⊢ LE.le 0 (Finset.univ.prod fun i => Norm.norm (m i))
      -/
      positivity
      /-
        🎉 no goals
      -/


@[simp]
theorem mkContinuousMultilinear_apply (f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)) {C : ℝ}
    (H : ∀ m₁ m₂, ‖f m₁ m₂‖ ≤ (C * ∏ i, ‖m₁ i‖) * ∏ i, ‖m₂ i‖) (m : ∀ i, E i) :
    ⇑(mkContinuousMultilinear f C H m) = f m :=
  rfl


theorem mkContinuousMultilinear_norm_le' (f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)) (C : ℝ)
    (H : ∀ m₁ m₂, ‖f m₁ m₂‖ ≤ (C * ∏ i, ‖m₁ i‖) * ∏ i, ‖m₂ i‖) :
    ‖mkContinuousMultilinear f C H‖ ≤ max C 0 := by
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    E : ι → Type wE
    E' : ι' → Type wE'
    G : Type wG
    inst✝⁸ : Fintype ι'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : Fintype ι
    inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
    f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
    C : Real
    H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
    ⊢ LE.le (Norm.norm (f.mkContinuousMultilinear C H)) (Max.max C 0)
  -/
  dsimp only [mkContinuousMultilinear]
  /-
    𝕜 : Type u
    ι : Type v
    ι' : Type v'
    E : ι → Type wE
    E' : ι' → Type wE'
    G : Type wG
    inst✝⁸ : Fintype ι'
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : SeminormedAddCommGroup G
    inst✝³ : NormedSpace 𝕜 G
    inst✝² : Fintype ι
    inst✝¹ : (i : ι') → SeminormedAddCommGroup (E' i)
    inst✝ : (i : ι') → NormedSpace 𝕜 (E' i)
    f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)
    C : Real
    H : ∀ (m₁ : (i : ι) → E i) (m₂ : (i : ι') → E' i), LE.le (Norm.norm ((f m₁) m₂ …
    ⊢ LE.le (Norm.norm ({ toFun := fun m => (f m).mkContinuous (HMul.hMul C (Finse …
  -/
  exact mkContinuous_norm_le _ (le_max_right _ _) _
  /-
    🎉 no goals
  -/


theorem mkContinuousMultilinear_norm_le (f : MultilinearMap 𝕜 E (MultilinearMap 𝕜 E' G)) {C : ℝ}
    (hC : 0 ≤ C) (H : ∀ m₁ m₂, ‖f m₁ m₂‖ ≤ (C * ∏ i, ‖m₁ i‖) * ∏ i, ‖m₂ i‖) :
    ‖mkContinuousMultilinear f C H‖ ≤ C :=
  (mkContinuousMultilinear_norm_le' f C H).trans_eq (max_eq_left hC)


theorem norm_compContinuousLinearMap_le (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i →L[𝕜] E₁ i) : ‖g.compContinuousLinearMap f‖ ≤ ‖g‖ * ∏ i, ‖f i‖ :=
                      /-
                        𝕜 : Type u
                        ι : Type v
                        E : ι → Type wE
                        E₁ : ι → Type wE₁
                        G : Type wG
                        inst✝⁷ : NontriviallyNormedField 𝕜
                        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
                        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
                        inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                        inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                        inst✝² : SeminormedAddCommGroup G
                        inst✝¹ : NormedSpace 𝕜 G
                        inst✝ : Fintype ι
                        g : ContinuousMultilinearMap 𝕜 E₁ G
                        f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
                        ⊢ LE.le 0 (HMul.hMul (Norm.norm g) (Finset.univ.prod fun i => Norm.norm (f i)))
                      -/
  opNorm_le_bound (by positivity) fun m =>
                      /-
                        🎉 no goals
                      -/
    calc
      ‖g fun i => f i (m i)‖ ≤ ‖g‖ * ∏ i, ‖f i (m i)‖ := g.le_opNorm _
      _ ≤ ‖g‖ * ∏ i, ‖f i‖ * ‖m i‖ :=
        (mul_le_mul_of_nonneg_left
          (prod_le_prod (fun _ _ => norm_nonneg _) fun i _ => (f i).le_opNorm (m i))
          (norm_nonneg g))
                                                /-
                                                  𝕜 : Type u
                                                  ι : Type v
                                                  E : ι → Type wE
                                                  E₁ : ι → Type wE₁
                                                  G : Type wG
                                                  inst✝⁷ : NontriviallyNormedField 𝕜
                                                  inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
                                                  inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
                                                  inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                                  inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                                  inst✝² : SeminormedAddCommGroup G
                                                  inst✝¹ : NormedSpace 𝕜 G
                                                  inst✝ : Fintype ι
                                                  g : ContinuousMultilinearMap 𝕜 E₁ G
                                                  f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
                                                  m : (i : ι) → E i
                                                  ⊢ Eq (HMul.hMul (Norm.norm g) (Finset.univ.prod fun i => HMul.hMul (Norm.norm  …
                                                -/
      _ = (‖g‖ * ∏ i, ‖f i‖) * ∏ i, ‖m i‖ := by rw [prod_mul_distrib, mul_assoc]
                                                /-
                                                  🎉 no goals
                                                -/


theorem norm_compContinuous_linearIsometry_le (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i →ₗᵢ[𝕜] E₁ i) :
    ‖g.compContinuousLinearMap fun i => (f i).toContinuousLinearMap‖ ≤ ‖g‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    E₁ : ι → Type wE₁
    G : Type wG
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    g : ContinuousMultilinearMap 𝕜 E₁ G
    f : (i : ι) → LinearIsometry (RingHom.id 𝕜) (E i) (E₁ i)
    ⊢ LE.le (Norm.norm (g.compContinuousLinearMap fun i => (f i).toContinuousLinea …
  -/
  refine opNorm_le_bound (norm_nonneg _) fun m => ?_
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    E₁ : ι → Type wE₁
    G : Type wG
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    g : ContinuousMultilinearMap 𝕜 E₁ G
    f : (i : ι) → LinearIsometry (RingHom.id 𝕜) (E i) (E₁ i)
    m : (i : ι) → E i
    ⊢ LE.le (Norm.norm ((g.compContinuousLinearMap fun i => (f i).toContinuousLine …
  -/
  apply (g.le_opNorm _).trans _
  simp only [ContinuousLinearMap.coe_coe, LinearIsometry.coe_toContinuousLinearMap,
    LinearIsometry.norm_map, le_rfl]


theorem norm_compContinuous_linearIsometryEquiv (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i ≃ₗᵢ[𝕜] E₁ i) :
    ‖g.compContinuousLinearMap fun i => (f i : E i →L[𝕜] E₁ i)‖ = ‖g‖ := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    E₁ : ι → Type wE₁
    G : Type wG
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    g : ContinuousMultilinearMap 𝕜 E₁ G
    f : (i : ι) → LinearIsometryEquiv (RingHom.id 𝕜) (E i) (E₁ i)
    ⊢ Eq (Norm.norm (g.compContinuousLinearMap fun i => ↑{ toLinearEquiv := (f i). …
  -/
  apply le_antisymm (g.norm_compContinuous_linearIsometry_le fun i => (f i).toLinearIsometry)
  have : g = (g.compContinuousLinearMap fun i => (f i : E i →L[𝕜] E₁ i)).compContinuousLinearMap
      fun i => ((f i).symm : E₁ i →L[𝕜] E i) := by
    ext1 m
    simp only [compContinuousLinearMap_apply, LinearIsometryEquiv.coe_coe'',
      LinearIsometryEquiv.apply_symm_apply]
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    E₁ : ι → Type wE₁
    G : Type wG
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
    inst✝² : SeminormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : Fintype ι
    g : ContinuousMultilinearMap 𝕜 E₁ G
    f : (i : ι) → LinearIsometryEquiv (RingHom.id 𝕜) (E i) (E₁ i)
    this : Eq g ((g.compContinuousLinearMap fun i => ↑{ toLinearEquiv := (f i).toL …
    ⊢ LE.le (Norm.norm g) (Norm.norm (g.compContinuousLinearMap fun i => (f i).toL …
  -/
  conv_lhs => rw [this]
  apply (g.compContinuousLinearMap fun i =>
    (f i : E i →L[𝕜] E₁ i)).norm_compContinuous_linearIsometry_le
      fun i => (f i).symm.toLinearIsometry


/-- `ContinuousMultilinearMap.compContinuousLinearMap` as a bundled continuous linear map.
This implementation fixes `f : Π i, E i →L[𝕜] E₁ i`.

Actually, the map is multilinear in `f`,
see `ContinuousMultilinearMap.compContinuousLinearMapContinuousMultilinear`.

For a version fixing `g` and varying `f`, see `compContinuousLinearMapLRight`. -/
def compContinuousLinearMapL (f : ∀ i, E i →L[𝕜] E₁ i) :
    ContinuousMultilinearMap 𝕜 E₁ G →L[𝕜] ContinuousMultilinearMap 𝕜 E G :=
  LinearMap.mkContinuous
    { toFun := fun g => g.compContinuousLinearMap f
      map_add' := fun _ _ => rfl
      map_smul' := fun _ _ => rfl }
    (∏ i, ‖f i‖)
    fun _ => (norm_compContinuousLinearMap_le _ _).trans_eq (mul_comm _ _)


@[simp]
theorem compContinuousLinearMapL_apply (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i →L[𝕜] E₁ i) : compContinuousLinearMapL f g = g.compContinuousLinearMap f :=
  rfl


variable (G) in
theorem norm_compContinuousLinearMapL_le (f : ∀ i, E i →L[𝕜] E₁ i) :
    ‖compContinuousLinearMapL (G := G) f‖ ≤ ∏ i, ‖f i‖ :=
                                       /-
                                         𝕜 : Type u
                                         ι : Type v
                                         E : ι → Type wE
                                         E₁ : ι → Type wE₁
                                         G : Type wG
                                         inst✝⁷ : NontriviallyNormedField 𝕜
                                         inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E i)
                                         inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E i)
                                         inst✝⁴ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                         inst✝³ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                         inst✝² : SeminormedAddCommGroup G
                                         inst✝¹ : NormedSpace 𝕜 G
                                         inst✝ : Fintype ι
                                         f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
                                         ⊢ LE.le 0 (Finset.univ.prod fun i => Norm.norm (f i))
                                       -/
  LinearMap.mkContinuous_norm_le _ (by positivity) _
                                       /-
                                         🎉 no goals
                                       -/


/-- `ContinuousMultilinearMap.compContinuousLinearMap` as a bundled continuous linear map.
This implementation fixes `g : ContinuousMultilinearMap 𝕜 E₁ G`.

Actually, the map is linear in `g`,
see `ContinuousMultilinearMap.compContinuousLinearMapContinuousMultilinear`.

For a version fixing `f` and varying `g`, see `compContinuousLinearMapL`. -/
def compContinuousLinearMapLRight (g : ContinuousMultilinearMap 𝕜 E₁ G) :
    ContinuousMultilinearMap 𝕜 (fun i ↦ E i →L[𝕜] E₁ i) (ContinuousMultilinearMap 𝕜 E G) :=
  MultilinearMap.mkContinuous
    { toFun := fun f => g.compContinuousLinearMap f
      map_update_add' := by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          ⊢ ∀ [inst : DecidableEq ι] (m : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) ( …
        -/
        intro h f i f₁ f₂
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          ⊢ Eq ((fun f => g.compContinuousLinearMap f) (Function.update f i (HAdd.hAdd f …
        -/
        ext x
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          x : (i : ι) → E i
          ⊢ Eq (((fun f => g.compContinuousLinearMap f) (Function.update f i (HAdd.hAdd  …
        -/
        simp only [compContinuousLinearMap_apply, add_apply]
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          x : (i : ι) → E i
          ⊢ Eq (g fun i_1 => (Function.update f i (HAdd.hAdd f₁ f₂) i_1) (x i_1)) (HAdd. …
        -/
        convert g.map_update_add (fun j ↦ f j (x j)) i (f₁ (x i)) (f₂ (x i)) <;>
          /-
            case h.e'_2.h.e'_6.h
            𝕜 : Type u
            ι : Type v
            ι' : Type v'
            E : ι → Type wE
            E₁ : ι → Type wE₁
            E' : ι' → Type wE'
            G : Type wG
            G' : Type wG'
            inst✝¹⁰ : Fintype ι'
            inst✝⁹ : NontriviallyNormedField 𝕜
            inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
            inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
            inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
            inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
            inst✝⁴ : SeminormedAddCommGroup G
            inst✝³ : NormedSpace 𝕜 G
            inst✝² : SeminormedAddCommGroup G'
            inst✝¹ : NormedSpace 𝕜 G'
            inst✝ : Fintype ι
            g : ContinuousMultilinearMap 𝕜 E₁ G
            h : DecidableEq ι
            f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
            i : ι
            f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
            x : (i : ι) → E i
            x✝ : ι
            ⊢ Eq ((Function.update f i (HAdd.hAdd f₁ f₂) x✝) (x x✝)) (Function.update (fun …
          -/
          /-
            🎉 no goals
          -/
          /-
            🎉 no goals
          -/
          exact apply_update (fun (i : ι) (f : E i →L[𝕜] E₁ i) ↦ f (x i)) f i _ _
          /-
            🎉 no goals
          -/
      map_update_smul' := by
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          ⊢ ∀ [inst : DecidableEq ι] (m : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) ( …
        -/
        intro h f i a f₀
        /-
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          a : 𝕜
          f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          ⊢ Eq ((fun f => g.compContinuousLinearMap f) (Function.update f i (HSMul.hSMul …
        -/
        ext x
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          a : 𝕜
          f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          x : (i : ι) → E i
          ⊢ Eq (((fun f => g.compContinuousLinearMap f) (Function.update f i (HSMul.hSMu …
        -/
        simp only [compContinuousLinearMap_apply, smul_apply]
        /-
          case H
          𝕜 : Type u
          ι : Type v
          ι' : Type v'
          E : ι → Type wE
          E₁ : ι → Type wE₁
          E' : ι' → Type wE'
          G : Type wG
          G' : Type wG'
          inst✝¹⁰ : Fintype ι'
          inst✝⁹ : NontriviallyNormedField 𝕜
          inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
          inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : SeminormedAddCommGroup G'
          inst✝¹ : NormedSpace 𝕜 G'
          inst✝ : Fintype ι
          g : ContinuousMultilinearMap 𝕜 E₁ G
          h : DecidableEq ι
          f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          i : ι
          a : 𝕜
          f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
          x : (i : ι) → E i
          ⊢ Eq (g fun i_1 => (Function.update f i (HSMul.hSMul a f₀) i_1) (x i_1)) (HSMu …
        -/
        convert g.map_update_smul (fun j ↦ f j (x j)) i a (f₀ (x i)) <;>
          /-
            case h.e'_2.h.e'_6.h
            𝕜 : Type u
            ι : Type v
            ι' : Type v'
            E : ι → Type wE
            E₁ : ι → Type wE₁
            E' : ι' → Type wE'
            G : Type wG
            G' : Type wG'
            inst✝¹⁰ : Fintype ι'
            inst✝⁹ : NontriviallyNormedField 𝕜
            inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
            inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
            inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
            inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
            inst✝⁴ : SeminormedAddCommGroup G
            inst✝³ : NormedSpace 𝕜 G
            inst✝² : SeminormedAddCommGroup G'
            inst✝¹ : NormedSpace 𝕜 G'
            inst✝ : Fintype ι
            g : ContinuousMultilinearMap 𝕜 E₁ G
            h : DecidableEq ι
            f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
            i : ι
            a : 𝕜
            f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
            x : (i : ι) → E i
            x✝ : ι
            ⊢ Eq ((Function.update f i (HSMul.hSMul a f₀) x✝) (x x✝)) (Function.update (fu …
          -/
          /-
            🎉 no goals
          -/
          exact apply_update (fun (i : ι) (f : E i →L[𝕜] E₁ i) ↦ f (x i)) f i _ _ }
          /-
            🎉 no goals
          -/
                      /-
                        𝕜 : Type u
                        ι : Type v
                        ι' : Type v'
                        E : ι → Type wE
                        E₁ : ι → Type wE₁
                        E' : ι' → Type wE'
                        G : Type wG
                        G' : Type wG'
                        inst✝¹⁰ : Fintype ι'
                        inst✝⁹ : NontriviallyNormedField 𝕜
                        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
                        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
                        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                        inst✝⁴ : SeminormedAddCommGroup G
                        inst✝³ : NormedSpace 𝕜 G
                        inst✝² : SeminormedAddCommGroup G'
                        inst✝¹ : NormedSpace 𝕜 G'
                        inst✝ : Fintype ι
                        g : ContinuousMultilinearMap 𝕜 E₁ G
                        f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
                        ⊢ LE.le (Norm.norm ({ toFun := fun f => g.compContinuousLinearMap f, map_updat …
                      -/
    (‖g‖) (fun f ↦ by simp [norm_compContinuousLinearMap_le])
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem compContinuousLinearMapLRight_apply (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i →L[𝕜] E₁ i) : compContinuousLinearMapLRight g f = g.compContinuousLinearMap f :=
  rfl


variable (E) in
theorem norm_compContinuousLinearMapLRight_le (g : ContinuousMultilinearMap 𝕜 E₁ G) :
    ‖compContinuousLinearMapLRight (E := E) g‖ ≤ ‖g‖ :=
  MultilinearMap.mkContinuous_norm_le _ (norm_nonneg _) _


open Function in
/-- If `f` is a collection of continuous linear maps, then the construction
`ContinuousMultilinearMap.compContinuousLinearMap`
sending a continuous multilinear map `g` to `g (f₁ ·, ..., fₙ ·)`
is continuous-linear in `g` and multilinear in `f₁, ..., fₙ`. -/
noncomputable def compContinuousLinearMapMultilinear :
    MultilinearMap 𝕜 (fun i ↦ E i →L[𝕜] E₁ i)
      ((ContinuousMultilinearMap 𝕜 E₁ G) →L[𝕜] ContinuousMultilinearMap 𝕜 E G) where
  toFun := compContinuousLinearMapL
  map_update_add' f i f₁ f₂ := by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹¹ : Fintype ι'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : SeminormedAddCommGroup G'
      inst✝² : NormedSpace 𝕜 G'
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      i : ι
      f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      ⊢ Eq (ContinuousMultilinearMap.compContinuousLinearMapL (Function.update f i ( …
    -/
    ext g x
    change (g fun j ↦ update f i (f₁ + f₂) j <| x j) =
        (g fun j ↦ update f i f₁ j <| x j) + g fun j ↦ update f i f₂ j (x j)
    /-
      case h.H
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹¹ : Fintype ι'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : SeminormedAddCommGroup G'
      inst✝² : NormedSpace 𝕜 G'
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      i : ι
      f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      g : ContinuousMultilinearMap 𝕜 E₁ G
      x : (i : ι) → E i
      ⊢ Eq (g fun j => (Function.update f i (HAdd.hAdd f₁ f₂) j) (x j)) (HAdd.hAdd ( …
    -/
    convert g.map_update_add (fun j ↦ f j (x j)) i (f₁ (x i)) (f₂ (x i)) <;>
      /-
        case h.e'_2.h.e'_6.h
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹¹ : Fintype ι'
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁵ : SeminormedAddCommGroup G
        inst✝⁴ : NormedSpace 𝕜 G
        inst✝³ : SeminormedAddCommGroup G'
        inst✝² : NormedSpace 𝕜 G'
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
        i : ι
        f₁ f₂ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
        g : ContinuousMultilinearMap 𝕜 E₁ G
        x : (i : ι) → E i
        x✝ : ι
        ⊢ Eq ((Function.update f i (HAdd.hAdd f₁ f₂) x✝) (x x✝)) (Function.update (fun …
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      exact apply_update (fun (i : ι) (f : E i →L[𝕜] E₁ i) ↦ f (x i)) f i _ _
      /-
        🎉 no goals
      -/
  map_update_smul' f i a f₀ := by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹¹ : Fintype ι'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : SeminormedAddCommGroup G'
      inst✝² : NormedSpace 𝕜 G'
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      i : ι
      a : 𝕜
      f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      ⊢ Eq (ContinuousMultilinearMap.compContinuousLinearMapL (Function.update f i ( …
    -/
    ext g x
    /-
      case h.H
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹¹ : Fintype ι'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : SeminormedAddCommGroup G'
      inst✝² : NormedSpace 𝕜 G'
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      i : ι
      a : 𝕜
      f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      g : ContinuousMultilinearMap 𝕜 E₁ G
      x : (i : ι) → E i
      ⊢ Eq (((ContinuousMultilinearMap.compContinuousLinearMapL (Function.update f i …
    -/
    change (g fun j ↦ update f i (a • f₀) j <| x j) = a • g fun j ↦ update f i f₀ j (x j)
    /-
      case h.H
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹¹ : Fintype ι'
      inst✝¹⁰ : NontriviallyNormedField 𝕜
      inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁵ : SeminormedAddCommGroup G
      inst✝⁴ : NormedSpace 𝕜 G
      inst✝³ : SeminormedAddCommGroup G'
      inst✝² : NormedSpace 𝕜 G'
      inst✝¹ : Fintype ι
      inst✝ : DecidableEq ι
      f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      i : ι
      a : 𝕜
      f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
      g : ContinuousMultilinearMap 𝕜 E₁ G
      x : (i : ι) → E i
      ⊢ Eq (g fun j => (Function.update f i (HSMul.hSMul a f₀) j) (x j)) (HSMul.hSMu …
    -/
    convert g.map_update_smul (fun j ↦ f j (x j)) i a (f₀ (x i)) <;>
      /-
        case h.e'_2.h.e'_6.h
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹¹ : Fintype ι'
        inst✝¹⁰ : NontriviallyNormedField 𝕜
        inst✝⁹ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁸ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁷ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁶ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁵ : SeminormedAddCommGroup G
        inst✝⁴ : NormedSpace 𝕜 G
        inst✝³ : SeminormedAddCommGroup G'
        inst✝² : NormedSpace 𝕜 G'
        inst✝¹ : Fintype ι
        inst✝ : DecidableEq ι
        f : (i : ι) → ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
        i : ι
        a : 𝕜
        f₀ : ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)
        g : ContinuousMultilinearMap 𝕜 E₁ G
        x : (i : ι) → E i
        x✝ : ι
        ⊢ Eq ((Function.update f i (HSMul.hSMul a f₀) x✝) (x x✝)) (Function.update (fu …
      -/
      /-
        🎉 no goals
      -/
      exact apply_update (fun (i : ι) (f : E i →L[𝕜] E₁ i) ↦ f (x i)) f i _ _
      /-
        🎉 no goals
      -/


/-- If `f` is a collection of continuous linear maps, then the construction
`ContinuousMultilinearMap.compContinuousLinearMap`
sending a continuous multilinear map `g` to `g (f₁ ·, ..., fₙ ·)` is continuous-linear in `g` and
continuous-multilinear in `f₁, ..., fₙ`. -/
noncomputable def compContinuousLinearMapContinuousMultilinear :
    ContinuousMultilinearMap 𝕜 (fun i ↦ E i →L[𝕜] E₁ i)
      ((ContinuousMultilinearMap 𝕜 E₁ G) →L[𝕜] ContinuousMultilinearMap 𝕜 E G) :=
  MultilinearMap.mkContinuous (𝕜 := 𝕜) (E := fun i ↦ E i →L[𝕜] E₁ i)
    (G := (ContinuousMultilinearMap 𝕜 E₁ G) →L[𝕜] ContinuousMultilinearMap 𝕜 E G)
    (compContinuousLinearMapMultilinear 𝕜 E E₁ G) 1 fun f ↦ by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → (fun i => ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)) i
        ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.compContinuousLinearMapMultiline …
      -/
      rw [one_mul]
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → (fun i => ContinuousLinearMap (RingHom.id 𝕜) (E i) (E₁ i)) i
        ⊢ LE.le (Norm.norm ((ContinuousMultilinearMap.compContinuousLinearMapMultiline …
      -/
      apply norm_compContinuousLinearMapL_le
      /-
        🎉 no goals
      -/


/-- `ContinuousMultilinearMap.compContinuousLinearMap` as a bundled continuous linear equiv,
given `f : Π i, E i ≃L[𝕜] E₁ i`. -/
def compContinuousLinearMapEquivL (f : ∀ i, E i ≃L[𝕜] E₁ i) :
    ContinuousMultilinearMap 𝕜 E₁ G ≃L[𝕜] ContinuousMultilinearMap 𝕜 E G :=
  { compContinuousLinearMapL fun i => (f i : E i →L[𝕜] E₁ i) with
    invFun := compContinuousLinearMapL fun i => ((f i).symm : E₁ i →L[𝕜] E i)
    continuous_toFun := (compContinuousLinearMapL fun i => (f i : E i →L[𝕜] E₁ i)).continuous
    continuous_invFun :=
      (compContinuousLinearMapL fun i => ((f i).symm : E₁ i →L[𝕜] E i)).continuous
    left_inv := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id 𝕜) (E i) (E₁ i)
        ⊢ Function.LeftInverse (⇑(ContinuousMultilinearMap.compContinuousLinearMapL fu …
      -/
      intro g
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id 𝕜) (E i) (E₁ i)
        g : ContinuousMultilinearMap 𝕜 E₁ G
        ⊢ Eq ((ContinuousMultilinearMap.compContinuousLinearMapL fun i => ↑(f i).symm) …
      -/
      ext1 m
      simp only [LinearMap.toFun_eq_coe, ContinuousLinearMap.coe_coe,
        compContinuousLinearMapL_apply, compContinuousLinearMap_apply,
        ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.apply_symm_apply]
    right_inv := by
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id 𝕜) (E i) (E₁ i)
        ⊢ Function.RightInverse (⇑(ContinuousMultilinearMap.compContinuousLinearMapL f …
      -/
      intro g
      /-
        𝕜 : Type u
        ι : Type v
        ι' : Type v'
        E : ι → Type wE
        E₁ : ι → Type wE₁
        E' : ι' → Type wE'
        G : Type wG
        G' : Type wG'
        inst✝¹⁰ : Fintype ι'
        inst✝⁹ : NontriviallyNormedField 𝕜
        inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E i)
        inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E i)
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : SeminormedAddCommGroup G'
        inst✝¹ : NormedSpace 𝕜 G'
        inst✝ : Fintype ι
        f : (i : ι) → ContinuousLinearEquiv (RingHom.id 𝕜) (E i) (E₁ i)
        g : ContinuousMultilinearMap 𝕜 E G
        ⊢ Eq ((↑__src✝).toFun ((ContinuousMultilinearMap.compContinuousLinearMapL fun  …
      -/
      ext1 m
      simp only [compContinuousLinearMapL_apply, LinearMap.toFun_eq_coe,
        ContinuousLinearMap.coe_coe, compContinuousLinearMap_apply,
        ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.symm_apply_apply] }


@[simp]
theorem compContinuousLinearMapEquivL_symm (f : ∀ i, E i ≃L[𝕜] E₁ i) :
    (compContinuousLinearMapEquivL G f).symm =
      compContinuousLinearMapEquivL G fun i : ι => (f i).symm :=
  rfl


@[simp]
theorem compContinuousLinearMapEquivL_apply (g : ContinuousMultilinearMap 𝕜 E₁ G)
    (f : ∀ i, E i ≃L[𝕜] E₁ i) :
    compContinuousLinearMapEquivL G f g =
      g.compContinuousLinearMap fun i => (f i : E i →L[𝕜] E₁ i) :=
  rfl


/-- One of the components of the iterated derivative of a continuous multilinear map. Given a
bijection `e` between a type `α` (typically `Fin k`) and a subset `s` of `ι`, this component is a
continuous multilinear map of `k` vectors `v₁, ..., vₖ`, mapping them
to `f (x₁, (v_{e.symm 2})₂, x₃, ...)`, where at indices `i` in `s` one uses the `i`-th coordinate of
the vector `v_{e.symm i}` and otherwise one uses the `i`-th coordinate of a reference vector `x`.
This is continuous multilinear in the components of `x` outside of `s`, and in the `v_j`. -/
noncomputable def iteratedFDerivComponent {α : Type*} [Fintype α]
    (f : ContinuousMultilinearMap 𝕜 E₁ G) {s : Set ι} (e : α ≃ s) [DecidablePred (· ∈ s)] :
    ContinuousMultilinearMap 𝕜 (fun (i : {a : ι // a ∉ s}) ↦ E₁ i)
      (ContinuousMultilinearMap 𝕜 (fun (_ : α) ↦ (∀ i, E₁ i)) G) :=
  (f.toMultilinearMap.iteratedFDerivComponent e).mkContinuousMultilinear ‖f‖ <| by
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹² : Fintype ι'
      inst✝¹¹ : NontriviallyNormedField 𝕜
      inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : NormedSpace 𝕜 G
      inst✝⁴ : SeminormedAddCommGroup G'
      inst✝³ : NormedSpace 𝕜 G'
      inst✝² : Fintype ι
      α : Type u_1
      inst✝¹ : Fintype α
      f : ContinuousMultilinearMap 𝕜 E₁ G
      s : Set ι
      e : Equiv α ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      ⊢ ∀ (m₁ : (i : Subtype fun a => Not (Membership.mem s a)) → E₁ ↑i) (m₂ : α → ( …
    -/
    intro x m
    simp only [MultilinearMap.iteratedFDerivComponent, MultilinearMap.domDomRestrictₗ,
      MultilinearMap.coe_mk, MultilinearMap.domDomRestrict_apply, coe_coe]
    /-
      𝕜 : Type u
      ι : Type v
      ι' : Type v'
      E : ι → Type wE
      E₁ : ι → Type wE₁
      E' : ι' → Type wE'
      G : Type wG
      G' : Type wG'
      inst✝¹² : Fintype ι'
      inst✝¹¹ : NontriviallyNormedField 𝕜
      inst✝¹⁰ : (i : ι) → SeminormedAddCommGroup (E i)
      inst✝⁹ : (i : ι) → NormedSpace 𝕜 (E i)
      inst✝⁸ : (i : ι) → SeminormedAddCommGroup (E₁ i)
      inst✝⁷ : (i : ι) → NormedSpace 𝕜 (E₁ i)
      inst✝⁶ : SeminormedAddCommGroup G
      inst✝⁵ : NormedSpace 𝕜 G
      inst✝⁴ : SeminormedAddCommGroup G'
      inst✝³ : NormedSpace 𝕜 G'
      inst✝² : Fintype ι
      α : Type u_1
      inst✝¹ : Fintype α
      f : ContinuousMultilinearMap 𝕜 E₁ G
      s : Set ι
      e : Equiv α ↑s
      inst✝ : DecidablePred fun x => Membership.mem s x
      x : (i : Subtype fun a => Not (Membership.mem s a)) → E₁ ↑i
      m : α → (i : ι) → E₁ i
      ⊢ LE.le (Norm.norm (f fun j => dite (Membership.mem s j) (fun h => m (e.symm ⟨ …
    -/
    apply (f.le_opNorm _).trans _
    classical
    rw [← prod_compl_mul_prod s.toFinset, mul_assoc]
    gcongr
    · apply le_of_eq
      have : ∀ x, x ∈ s.toFinsetᶜ ↔ (fun x ↦ x ∉ s) x := by simp
      rw [prod_subtype _ this]
      congr with i
      simp [i.2]
    · rw [prod_subtype _ (fun _ ↦ s.mem_toFinset), ← Equiv.prod_comp e.symm]
      apply Finset.prod_le_prod (fun i _ ↦ norm_nonneg _) (fun i _ ↦ ?_)
      simpa only [i.2, ↓reduceDIte, Subtype.coe_eta] using norm_le_pi_norm (m (e.symm i)) ↑i


@[simp] lemma iteratedFDerivComponent_apply {α : Type*} [Fintype α]
    (f : ContinuousMultilinearMap 𝕜 E₁ G) {s : Set ι} (e : α ≃ s) [DecidablePred (· ∈ s)]
    (v : ∀ i : {a : ι // a ∉ s}, E₁ i) (w : α → (∀ i, E₁ i)) :
    f.iteratedFDerivComponent e v w =
      f (fun j ↦ if h : j ∈ s then w (e.symm ⟨j, h⟩) j else v ⟨j, h⟩) := by
  simp [iteratedFDerivComponent, MultilinearMap.iteratedFDerivComponent,
    MultilinearMap.domDomRestrictₗ]


lemma norm_iteratedFDerivComponent_le {α : Type*} [Fintype α]
    (f : ContinuousMultilinearMap 𝕜 E₁ G) {s : Set ι} (e : α ≃ s) [DecidablePred (· ∈ s)]
    (x : (i : ι) → E₁ i) :
    ‖f.iteratedFDerivComponent e (x ·)‖ ≤ ‖f‖ * ‖x‖ ^ (Fintype.card ι - Fintype.card α) := calc
  ‖f.iteratedFDerivComponent e (fun i ↦ x i)‖
    ≤ ‖f.iteratedFDerivComponent e‖ * ∏ i : {a : ι // a ∉ s}, ‖x i‖ :=
      ContinuousMultilinearMap.le_opNorm _ _
  _ ≤ ‖f‖ * ∏ _i : {a : ι // a ∉ s}, ‖x‖ := by
      /-
        𝕜 : Type u
        ι : Type v
        E₁ : ι → Type wE₁
        G : Type wG
        inst✝⁷ : NontriviallyNormedField 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
        inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
        inst✝⁴ : SeminormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        inst✝² : Fintype ι
        α : Type u_1
        inst✝¹ : Fintype α
        f : ContinuousMultilinearMap 𝕜 E₁ G
        s : Set ι
        e : Equiv α ↑s
        inst✝ : DecidablePred fun x => Membership.mem s x
        x : (i : ι) → E₁ i
        ⊢ LE.le (HMul.hMul (Norm.norm (f.iteratedFDerivComponent e)) (Finset.univ.prod …
      -/
      gcongr
        /-
          case h₁
          𝕜 : Type u
          ι : Type v
          E₁ : ι → Type wE₁
          G : Type wG
          inst✝⁷ : NontriviallyNormedField 𝕜
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : Fintype ι
          α : Type u_1
          inst✝¹ : Fintype α
          f : ContinuousMultilinearMap 𝕜 E₁ G
          s : Set ι
          e : Equiv α ↑s
          inst✝ : DecidablePred fun x => Membership.mem s x
          x : (i : ι) → E₁ i
          ⊢ LE.le (Norm.norm (f.iteratedFDerivComponent e)) (Norm.norm f)
        -/
      · exact MultilinearMap.mkContinuousMultilinear_norm_le _ (norm_nonneg _) _
        /-
          🎉 no goals
        -/
        /-
          case h₂.h1
          𝕜 : Type u
          ι : Type v
          E₁ : ι → Type wE₁
          G : Type wG
          inst✝⁷ : NontriviallyNormedField 𝕜
          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
          inst✝⁴ : SeminormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          inst✝² : Fintype ι
          α : Type u_1
          inst✝¹ : Fintype α
          f : ContinuousMultilinearMap 𝕜 E₁ G
          s : Set ι
          e : Equiv α ↑s
          inst✝ : DecidablePred fun x => Membership.mem s x
          x : (i : ι) → E₁ i
          i✝ : Subtype fun a => Not (Membership.mem s a)
          a✝ : Membership.mem Finset.univ i✝
          ⊢ LE.le (Norm.norm (x ↑i✝)) (Norm.norm x)
        -/
      · exact norm_le_pi_norm _ _
        /-
          🎉 no goals
        -/
                                                        /-
                                                          𝕜 : Type u
                                                          ι : Type v
                                                          E₁ : ι → Type wE₁
                                                          G : Type wG
                                                          inst✝⁷ : NontriviallyNormedField 𝕜
                                                          inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                                          inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                                          inst✝⁴ : SeminormedAddCommGroup G
                                                          inst✝³ : NormedSpace 𝕜 G
                                                          inst✝² : Fintype ι
                                                          α : Type u_1
                                                          inst✝¹ : Fintype α
                                                          f : ContinuousMultilinearMap 𝕜 E₁ G
                                                          s : Set ι
                                                          e : Equiv α ↑s
                                                          inst✝ : DecidablePred fun x => Membership.mem s x
                                                          x : (i : ι) → E₁ i
                                                          ⊢ Eq (HMul.hMul (Norm.norm f) (Finset.univ.prod fun _i => Norm.norm x)) (HMul. …
                                                        -/
  _ = ‖f‖ * ‖x‖ ^ (Fintype.card {a : ι // a ∉ s}) := by rw [prod_const, card_univ]
                                                        /-
                                                          🎉 no goals
                                                        -/
                                                          /-
                                                            𝕜 : Type u
                                                            ι : Type v
                                                            E₁ : ι → Type wE₁
                                                            G : Type wG
                                                            inst✝⁷ : NontriviallyNormedField 𝕜
                                                            inst✝⁶ : (i : ι) → SeminormedAddCommGroup (E₁ i)
                                                            inst✝⁵ : (i : ι) → NormedSpace 𝕜 (E₁ i)
                                                            inst✝⁴ : SeminormedAddCommGroup G
                                                            inst✝³ : NormedSpace 𝕜 G
                                                            inst✝² : Fintype ι
                                                            α : Type u_1
                                                            inst✝¹ : Fintype α
                                                            f : ContinuousMultilinearMap 𝕜 E₁ G
                                                            s : Set ι
                                                            e : Equiv α ↑s
                                                            inst✝ : DecidablePred fun x => Membership.mem s x
                                                            x : (i : ι) → E₁ i
                                                            ⊢ Eq (HMul.hMul (Norm.norm f) (HPow.hPow (Norm.norm x) (Fintype.card (Subtype  …
                                                          -/
  _ = ‖f‖ * ‖x‖ ^ (Fintype.card ι - Fintype.card α) := by simp [Fintype.card_congr e]
                                                          /-
                                                            🎉 no goals
                                                          -/


open Classical in
/-- The `k`-th iterated derivative of a continuous multilinear map `f` at the point `x`. It is a
continuous multilinear map of `k` vectors `v₁, ..., vₖ` (with the same type as `x`), mapping them
to `∑ f (x₁, (v_{i₁})₂, x₃, ...)`, where at each index `j` one uses either `xⱼ` or one
of the `(vᵢ)ⱼ`, and each `vᵢ` has to be used exactly once.
The sum is parameterized by the embeddings of `Fin k` in the index type `ι` (or, equivalently,
by the subsets `s` of `ι` of cardinality `k` and then the bijections between `Fin k` and `s`).

The fact that this is indeed the iterated Fréchet derivative is proved in
`ContinuousMultilinearMap.iteratedFDeriv_eq`.
-/
protected def iteratedFDeriv (f : ContinuousMultilinearMap 𝕜 E₁ G) (k : ℕ) (x : (i : ι) → E₁ i) :
    ContinuousMultilinearMap 𝕜 (fun (_ : Fin k) ↦ (∀ i, E₁ i)) G :=
  ∑ e : Fin k ↪ ι, iteratedFDerivComponent f e.toEquivRange (Pi.compRightL 𝕜 _ Subtype.val x)


/-- Controlling the norm of `f.iteratedFDeriv` when `f` is continuous multilinear. For the same
bound on the iterated derivative of `f` in the calculus sense,
see `ContinuousMultilinearMap.norm_iteratedFDeriv_le`. -/
lemma norm_iteratedFDeriv_le' (f : ContinuousMultilinearMap 𝕜 E₁ G) (k : ℕ) (x : (i : ι) → E₁ i) :
    ‖f.iteratedFDeriv k x‖
      ≤ Nat.descFactorial (Fintype.card ι) k * ‖f‖ * ‖x‖ ^ (Fintype.card ι - k) := by
  classical
  calc ‖f.iteratedFDeriv k x‖
  _ ≤ ∑ e : Fin k ↪ ι, ‖iteratedFDerivComponent f e.toEquivRange (fun i ↦ x i)‖ := norm_sum_le _ _
  _ ≤ ∑ _ : Fin k ↪ ι, ‖f‖ * ‖x‖ ^ (Fintype.card ι - k) := by
    gcongr with e _
    simpa using norm_iteratedFDerivComponent_le f e.toEquivRange x
  _ = Nat.descFactorial (Fintype.card ι) k * ‖f‖ * ‖x‖ ^ (Fintype.card ι - k) := by
    simp [card_univ, mul_assoc]


/-- A continuous linear map is zero iff its norm vanishes. -/
theorem opNorm_zero_iff {f : ContinuousMultilinearMap 𝕜 E G} : ‖f‖ = 0 ↔ f = 0 := by
  /-
    𝕜 : Type u
    ι : Type v
    E : ι → Type wE
    G : Type wG
    inst✝⁵ : Fintype ι
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : (i : ι) → SeminormedAddCommGroup (E i)
    inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : ContinuousMultilinearMap 𝕜 E G
    ⊢ Iff (Eq (Norm.norm f) 0) (Eq f 0)
  -/
  simp [← (opNorm_nonneg f).le_iff_eq, opNorm_le_iff le_rfl, ContinuousMultilinearMap.ext_iff]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-02")] alias op_norm_zero_iff := opNorm_zero_iff


/-- Continuous multilinear maps themselves form a normed group with respect to
    the operator norm. -/
instance normedAddCommGroup : NormedAddCommGroup (ContinuousMultilinearMap 𝕜 E G) :=
  NormedAddCommGroup.ofSeparation fun _ ↦ opNorm_zero_iff.mp


/-- An alias of `ContinuousMultilinearMap.normedAddCommGroup` with non-dependent types to help
typeclass search. -/
instance normedAddCommGroup' :
    NormedAddCommGroup (ContinuousMultilinearMap 𝕜 (fun _ : ι => G') G) :=
  ContinuousMultilinearMap.normedAddCommGroup


theorem norm_ofSubsingleton_id [Subsingleton ι] [Nontrivial G] (i : ι) :
    ‖ofSubsingleton 𝕜 G G i (.id _ _)‖ = 1 := by
  /-
    𝕜 : Type u
    ι : Type v
    G : Type wG
    inst✝⁵ : Fintype ι
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup G
    inst✝² : NormedSpace 𝕜 G
    inst✝¹ : Subsingleton ι
    inst✝ : Nontrivial G
    i : ι
    ⊢ Eq (Norm.norm ((ContinuousMultilinearMap.ofSubsingleton 𝕜 G G i) (Continuous …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nnnorm_ofSubsingleton_id [Subsingleton ι] [Nontrivial G] (i : ι) :
    ‖ofSubsingleton 𝕜 G G i (.id _ _)‖₊ = 1 :=
  NNReal.eq <| norm_ofSubsingleton_id ..


/-- If a multilinear map in finitely many variables on normed spaces satisfies the inequality
`‖f m‖ ≤ C * ∏ i, ‖m i‖` on a shell `ε i / ‖c i‖ < ‖m i‖ < ε i` for some positive numbers `ε i`
and elements `c i : 𝕜`, `1 < ‖c i‖`, then it satisfies this inequality for all `m`. -/
theorem bound_of_shell (f : MultilinearMap 𝕜 E G) {ε : ι → ℝ} {C : ℝ} {c : ι → 𝕜}
    (hε : ∀ i, 0 < ε i) (hc : ∀ i, 1 < ‖c i‖)
    (hf : ∀ m : ∀ i, E i, (∀ i, ε i / ‖c i‖ ≤ ‖m i‖) → (∀ i, ‖m i‖ < ε i) → ‖f m‖ ≤ C * ∏ i, ‖m i‖)
    (m : ∀ i, E i) : ‖f m‖ ≤ C * ∏ i, ‖m i‖ :=
  bound_of_shell_of_norm_map_coord_zero f
                /-
                  𝕜 : Type u
                  ι : Type v
                  E : ι → Type wE
                  G : Type wG
                  inst✝⁵ : Fintype ι
                  inst✝⁴ : NontriviallyNormedField 𝕜
                  inst✝³ : (i : ι) → NormedAddCommGroup (E i)
                  inst✝² : (i : ι) → NormedSpace 𝕜 (E i)
                  inst✝¹ : SeminormedAddCommGroup G
                  inst✝ : NormedSpace 𝕜 G
                  f : MultilinearMap 𝕜 E G
                  ε : ι → Real
                  C : Real
                  c : ι → 𝕜
                  hε : ∀ (i : ι), LT.lt 0 (ε i)
                  hc : ∀ (i : ι), LT.lt 1 (Norm.norm (c i))
                  hf : ∀ (m : (i : ι) → E i), (∀ (i : ι), LE.le (HDiv.hDiv (ε i) (Norm.norm (c i …
                  m m✝ : (i : ι) → E i
                  i✝ : ι
                  h : Eq (Norm.norm (m✝ i✝)) 0
                  ⊢ Eq (Norm.norm (f m✝)) 0
                -/
    (fun h ↦ by rw [map_coord_zero f _ (norm_eq_zero.1 h), norm_zero]) hε hc hf m
                /-
                  🎉 no goals
                -/


