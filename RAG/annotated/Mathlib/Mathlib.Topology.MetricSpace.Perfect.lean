private theorem Perfect.small_diam_aux (hC : Perfect C) (ε_pos : 0 < ε) {x : α} (xC : x ∈ C) :
    let D := closure (EMetric.ball x (ε / 2) ∩ C)
    Perfect D ∧ D.Nonempty ∧ D ⊆ C ∧ EMetric.diam D ≤ ε := by
  have : x ∈ EMetric.ball x (ε / 2) := by
    apply EMetric.mem_ball_self
    rw [ENNReal.div_pos_iff]
    exact ⟨ne_of_gt ε_pos, by norm_num⟩
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    ⊢ let D := closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C);
      And (Perfect D) (And D.Nonempty (And (HasSubset.Subset D C) (LE.le (EMetric. …
  -/
  have := hC.closure_nhds_inter x xC this EMetric.isOpen_ball
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
    ⊢ let D := closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C);
      And (Perfect D) (And D.Nonempty (And (HasSubset.Subset D C) (LE.le (EMetric. …
  -/
  refine ⟨this.1, this.2, ?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : MetricSpace α
      C : Set α
      ε : ENNReal
      hC : Perfect C
      ε_pos : LT.lt 0 ε
      x : α
      xC : Membership.mem C x
      this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
      this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
      ⊢ HasSubset.Subset (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) C
    -/
  · rw [IsClosed.closure_subset_iff hC.closed]
    /-
      case refine_1
      α : Type u_1
      inst✝ : MetricSpace α
      C : Set α
      ε : ENNReal
      hC : Perfect C
      ε_pos : LT.lt 0 ε
      x : α
      xC : Membership.mem C x
      this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
      this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
      ⊢ HasSubset.Subset (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C) C
    -/
    apply inter_subset_right
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
    ⊢ LE.le (EMetric.diam (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C …
  -/
  rw [EMetric.diam_closure]
  /-
    case refine_2
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
    ⊢ LE.le (EMetric.diam (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) ε
  -/
  apply le_trans (EMetric.diam_mono inter_subset_left)
  /-
    case refine_2
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
    ⊢ LE.le (EMetric.diam (EMetric.ball x (HDiv.hDiv ε 2))) ε
  -/
  convert EMetric.diam_ball (x := x)
  /-
    case h.e'_4
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    ε_pos : LT.lt 0 ε
    x : α
    xC : Membership.mem C x
    this✝ : Membership.mem (EMetric.ball x (HDiv.hDiv ε 2)) x
    this : And (Perfect (closure (Inter.inter (EMetric.ball x (HDiv.hDiv ε 2)) C)) …
    ⊢ Eq ε (HMul.hMul 2 (HDiv.hDiv ε 2))
  -/
                                            /-
                                              🎉 no goals
                                            -/
  rw [mul_comm, ENNReal.div_mul_cancel] <;> norm_num
                                            /-
                                              🎉 no goals
                                            -/


/-- A refinement of `Perfect.splitting` for metric spaces, where we also control
the diameter of the new perfect sets. -/
theorem Perfect.small_diam_splitting (hC : Perfect C) (hnonempty : C.Nonempty) (ε_pos : 0 < ε) :
    ∃ C₀ C₁ : Set α, (Perfect C₀ ∧ C₀.Nonempty ∧ C₀ ⊆ C ∧ EMetric.diam C₀ ≤ ε) ∧
    (Perfect C₁ ∧ C₁.Nonempty ∧ C₁ ⊆ C ∧ EMetric.diam C₁ ≤ ε) ∧ Disjoint C₀ C₁ := by
  /-
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (An …
  -/
  rcases hC.splitting hnonempty with ⟨D₀, D₁, ⟨perf0, non0, sub0⟩, ⟨perf1, non1, sub1⟩, hdisj⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    D₀ D₁ : Set α
    perf0 : Perfect D₀
    non0 : D₀.Nonempty
    sub0 : HasSubset.Subset D₀ C
    hdisj : Disjoint D₀ D₁
    perf1 : Perfect D₁
    non1 : D₁.Nonempty
    sub1 : HasSubset.Subset D₁ C
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (An …
  -/
  cases' non0 with x₀ hx₀
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    D₀ D₁ : Set α
    perf0 : Perfect D₀
    sub0 : HasSubset.Subset D₀ C
    hdisj : Disjoint D₀ D₁
    perf1 : Perfect D₁
    non1 : D₁.Nonempty
    sub1 : HasSubset.Subset D₁ C
    x₀ : α
    hx₀ : Membership.mem D₀ x₀
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (An …
  -/
  cases' non1 with x₁ hx₁
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    D₀ D₁ : Set α
    perf0 : Perfect D₀
    sub0 : HasSubset.Subset D₀ C
    hdisj : Disjoint D₀ D₁
    perf1 : Perfect D₁
    sub1 : HasSubset.Subset D₁ C
    x₀ : α
    hx₀ : Membership.mem D₀ x₀
    x₁ : α
    hx₁ : Membership.mem D₁ x₁
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (An …
  -/
  rcases perf0.small_diam_aux ε_pos hx₀ with ⟨perf0', non0', sub0', diam0⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    D₀ D₁ : Set α
    perf0 : Perfect D₀
    sub0 : HasSubset.Subset D₀ C
    hdisj : Disjoint D₀ D₁
    perf1 : Perfect D₁
    sub1 : HasSubset.Subset D₁ C
    x₀ : α
    hx₀ : Membership.mem D₀ x₀
    x₁ : α
    hx₁ : Membership.mem D₁ x₁
    perf0' : Perfect (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2)) D₀))
    non0' : (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2)) D₀)).Nonempty
    sub0' : HasSubset.Subset (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2 …
    diam0 : LE.le (EMetric.diam (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv  …
    ⊢ Exists fun C₀ => Exists fun C₁ => And (And (Perfect C₀) (And C₀.Nonempty (An …
  -/
  rcases perf1.small_diam_aux ε_pos hx₁ with ⟨perf1', non1', sub1', diam1⟩
  refine
    ⟨closure (EMetric.ball x₀ (ε / 2) ∩ D₀), closure (EMetric.ball x₁ (ε / 2) ∩ D₁),
      ⟨perf0', non0', sub0'.trans sub0, diam0⟩, ⟨perf1', non1', sub1'.trans sub1, diam1⟩, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.intro.i …
    α : Type u_1
    inst✝ : MetricSpace α
    C : Set α
    ε : ENNReal
    hC : Perfect C
    hnonempty : C.Nonempty
    ε_pos : LT.lt 0 ε
    D₀ D₁ : Set α
    perf0 : Perfect D₀
    sub0 : HasSubset.Subset D₀ C
    hdisj : Disjoint D₀ D₁
    perf1 : Perfect D₁
    sub1 : HasSubset.Subset D₁ C
    x₀ : α
    hx₀ : Membership.mem D₀ x₀
    x₁ : α
    hx₁ : Membership.mem D₁ x₁
    perf0' : Perfect (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2)) D₀))
    non0' : (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2)) D₀)).Nonempty
    sub0' : HasSubset.Subset (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2 …
    diam0 : LE.le (EMetric.diam (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv  …
    perf1' : Perfect (closure (Inter.inter (EMetric.ball x₁ (HDiv.hDiv ε 2)) D₁))
    non1' : (closure (Inter.inter (EMetric.ball x₁ (HDiv.hDiv ε 2)) D₁)).Nonempty
    sub1' : HasSubset.Subset (closure (Inter.inter (EMetric.ball x₁ (HDiv.hDiv ε 2 …
    diam1 : LE.le (EMetric.diam (closure (Inter.inter (EMetric.ball x₁ (HDiv.hDiv  …
    ⊢ Disjoint (closure (Inter.inter (EMetric.ball x₀ (HDiv.hDiv ε 2)) D₀)) (closu …
  -/
                                    /-
                                      🎉 no goals
                                    -/
  apply Disjoint.mono _ _ hdisj <;> assumption
                                    /-
                                      🎉 no goals
                                    -/


/-- Any nonempty perfect set in a complete metric space admits a continuous injection
from the Cantor space, `ℕ → Bool`. -/
theorem Perfect.exists_nat_bool_injection
    (hC : Perfect C) (hnonempty : C.Nonempty) [CompleteSpace α] :
    ∃ f : (ℕ → Bool) → α, range f ⊆ C ∧ Continuous f ∧ Injective f := by
  /-
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  obtain ⟨u, -, upos', hu⟩ := exists_seq_strictAnti_tendsto' (zero_lt_one' ℝ≥0∞)
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  have upos := fun n => (upos' n).1
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    upos : ∀ (n : Nat), LT.lt 0 (u n)
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  let P := Subtype fun E : Set α => Perfect E ∧ E.Nonempty
  choose C0 C1 h0 h1 hdisj using
    fun {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ℝ≥0∞} (hε : 0 < ε) =>
    hC.small_diam_splitting hnonempty hε
  let DP : List Bool → P := fun l => by
    induction' l with a l ih; · exact ⟨C, ⟨hC, hnonempty⟩⟩
    cases a
    · use C0 ih.property.1 ih.property.2 (upos (l.length + 1))
      exact ⟨(h0 _ _ _).1, (h0 _ _ _).2.1⟩
    use C1 ih.property.1 ih.property.2 (upos (l.length + 1))
    exact ⟨(h1 _ _ _).1, (h1 _ _ _).2.1⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    upos : ∀ (n : Nat), LT.lt 0 (u n)
    P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
    C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
    h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
    DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  let D : List Bool → Set α := fun l => (DP l).val
  have hanti : ClosureAntitone D := by
    refine Antitone.closureAntitone ?_ fun l => (DP l).property.1.closed
    intro l a
    cases a
    · exact (h0 _ _ _).2.2.1
    exact (h1 _ _ _).2.2.1
  have hdiam : VanishingDiam D := by
    intro x
    apply tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hu
    · simp
    rw [eventually_atTop]
    refine ⟨1, fun m (hm : 1 ≤ m) => ?_⟩
    rw [Nat.one_le_iff_ne_zero] at hm
    rcases Nat.exists_eq_succ_of_ne_zero hm with ⟨n, rfl⟩
    dsimp
    cases x n
    · convert (h0 _ _ _).2.2.2
      rw [PiNat.res_length]
    convert (h1 _ _ _).2.2.2
    rw [PiNat.res_length]
  have hdisj' : CantorScheme.Disjoint D := by
    rintro l (a | a) (b | b) hab <;> try contradiction
    · exact hdisj _ _ _
    exact (hdisj _ _ _).symm
  have hdom : ∀ {x : ℕ → Bool}, x ∈ (inducedMap D).1 := fun {x} => by
    rw [hanti.map_of_vanishingDiam hdiam fun l => (DP l).property.2]
    apply mem_univ
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    upos : ∀ (n : Nat), LT.lt 0 (u n)
    P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
    C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
    h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
    DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
    D : List Bool → Set α := fun l => ↑(DP l)
    hanti : CantorScheme.ClosureAntitone D
    hdiam : CantorScheme.VanishingDiam D
    hdisj' : CantorScheme.Disjoint D
    hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  refine ⟨fun x => (inducedMap D).2 ⟨x, hdom⟩, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      α : Type u_1
      inst✝¹ : MetricSpace α
      C : Set α
      hC : Perfect C
      hnonempty : C.Nonempty
      inst✝ : CompleteSpace α
      u : Nat → ENNReal
      upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      upos : ∀ (n : Nat), LT.lt 0 (u n)
      P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
      C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
      h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
      DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
      D : List Bool → Set α := fun l => ↑(DP l)
      hanti : CantorScheme.ClosureAntitone D
      hdiam : CantorScheme.VanishingDiam D
      hdisj' : CantorScheme.Disjoint D
      hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
      ⊢ HasSubset.Subset (Set.range fun x => (CantorScheme.inducedMap D).snd ⟨x, ⋯⟩) C
    -/
  · rintro y ⟨x, rfl⟩
    /-
      case intro.intro.intro.refine_1.intro
      α : Type u_1
      inst✝¹ : MetricSpace α
      C : Set α
      hC : Perfect C
      hnonempty : C.Nonempty
      inst✝ : CompleteSpace α
      u : Nat → ENNReal
      upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      upos : ∀ (n : Nat), LT.lt 0 (u n)
      P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
      C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
      h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
      DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
      D : List Bool → Set α := fun l => ↑(DP l)
      hanti : CantorScheme.ClosureAntitone D
      hdiam : CantorScheme.VanishingDiam D
      hdisj' : CantorScheme.Disjoint D
      hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
      x : Nat → Bool
      ⊢ Membership.mem C ((fun x => (CantorScheme.inducedMap D).snd ⟨x, ⋯⟩) x)
    -/
    exact map_mem ⟨_, hdom⟩ 0
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : MetricSpace α
      C : Set α
      hC : Perfect C
      hnonempty : C.Nonempty
      inst✝ : CompleteSpace α
      u : Nat → ENNReal
      upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      upos : ∀ (n : Nat), LT.lt 0 (u n)
      P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
      C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
      h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
      DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
      D : List Bool → Set α := fun l => ↑(DP l)
      hanti : CantorScheme.ClosureAntitone D
      hdiam : CantorScheme.VanishingDiam D
      hdisj' : CantorScheme.Disjoint D
      hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
      ⊢ Continuous fun x => (CantorScheme.inducedMap D).snd ⟨x, ⋯⟩
    -/
  · apply hdiam.map_continuous.comp
    /-
      case intro.intro.intro.refine_2
      α : Type u_1
      inst✝¹ : MetricSpace α
      C : Set α
      hC : Perfect C
      hnonempty : C.Nonempty
      inst✝ : CompleteSpace α
      u : Nat → ENNReal
      upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      upos : ∀ (n : Nat), LT.lt 0 (u n)
      P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
      C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
      h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
      hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
      DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
      D : List Bool → Set α := fun l => ↑(DP l)
      hanti : CantorScheme.ClosureAntitone D
      hdiam : CantorScheme.VanishingDiam D
      hdisj' : CantorScheme.Disjoint D
      hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
      ⊢ Continuous fun x => ⟨x, ⋯⟩
    -/
    continuity
    /-
      🎉 no goals
    -/
  /-
    case intro.intro.intro.refine_3
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    upos : ∀ (n : Nat), LT.lt 0 (u n)
    P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
    C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
    h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
    DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
    D : List Bool → Set α := fun l => ↑(DP l)
    hanti : CantorScheme.ClosureAntitone D
    hdiam : CantorScheme.VanishingDiam D
    hdisj' : CantorScheme.Disjoint D
    hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
    ⊢ Function.Injective fun x => (CantorScheme.inducedMap D).snd ⟨x, ⋯⟩
  -/
  intro x y hxy
  /-
    case intro.intro.intro.refine_3
    α : Type u_1
    inst✝¹ : MetricSpace α
    C : Set α
    hC : Perfect C
    hnonempty : C.Nonempty
    inst✝ : CompleteSpace α
    u : Nat → ENNReal
    upos' : ∀ (n : Nat), Membership.mem (Set.Ioo 0 1) (u n)
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    upos : ∀ (n : Nat), LT.lt 0 (u n)
    P : Type (max 0 u_1) := Subtype fun E => And (Perfect E) E.Nonempty
    C0 C1 : {C : Set α} → Perfect C → C.Nonempty → {ε : ENNReal} → LT.lt 0 ε → Set α
    h0 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    h1 : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal} (hε …
    hdisj : ∀ {C : Set α} (hC : Perfect C) (hnonempty : C.Nonempty) {ε : ENNReal}  …
    DP : List Bool → P := fun l => List.rec ⟨C, ⋯⟩ (fun a l ih => Bool.casesOn (mo …
    D : List Bool → Set α := fun l => ↑(DP l)
    hanti : CantorScheme.ClosureAntitone D
    hdiam : CantorScheme.VanishingDiam D
    hdisj' : CantorScheme.Disjoint D
    hdom : ∀ {x : Nat → Bool}, Membership.mem (CantorScheme.inducedMap D).fst x
    x y : Nat → Bool
    hxy : Eq ((fun x => (CantorScheme.inducedMap D).snd ⟨x, ⋯⟩) x) ((fun x => (Can …
    ⊢ Eq x y
  -/
  simpa only [← Subtype.val_inj] using hdisj'.map_injective hxy
  /-
    🎉 no goals
  -/


/-- Any closed uncountable subset of a Polish space admits a continuous injection
from the Cantor space `ℕ → Bool`. -/
theorem IsClosed.exists_nat_bool_injection_of_not_countable {α : Type*} [TopologicalSpace α]
    [PolishSpace α] {C : Set α} (hC : IsClosed C) (hunc : ¬C.Countable) :
    ∃ f : (ℕ → Bool) → α, range f ⊆ C ∧ Continuous f ∧ Function.Injective f := by
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    C : Set α
    hC : IsClosed C
    hunc : Not C.Countable
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  letI := upgradePolishSpace α
  /-
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    C : Set α
    hC : IsClosed C
    hunc : Not C.Countable
    this : UpgradedPolishSpace α := upgradePolishSpace α
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  obtain ⟨D, hD, Dnonempty, hDC⟩ := exists_perfect_nonempty_of_isClosed_of_not_countable hC hunc
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    C : Set α
    hC : IsClosed C
    hunc : Not C.Countable
    this : UpgradedPolishSpace α := upgradePolishSpace α
    D : Set α
    hD : Perfect D
    Dnonempty : D.Nonempty
    hDC : HasSubset.Subset D C
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  obtain ⟨f, hfD, hf⟩ := hD.exists_nat_bool_injection Dnonempty
  /-
    case intro.intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : TopologicalSpace α
    inst✝ : PolishSpace α
    C : Set α
    hC : IsClosed C
    hunc : Not C.Countable
    this : UpgradedPolishSpace α := upgradePolishSpace α
    D : Set α
    hD : Perfect D
    Dnonempty : D.Nonempty
    hDC : HasSubset.Subset D C
    f : (Nat → Bool) → α
    hfD : HasSubset.Subset (Set.range f) D
    hf : And (Continuous f) (Function.Injective f)
    ⊢ Exists fun f => And (HasSubset.Subset (Set.range f) C) (And (Continuous f) ( …
  -/
  exact ⟨f, hfD.trans hDC, hf⟩
  /-
    🎉 no goals
  -/

