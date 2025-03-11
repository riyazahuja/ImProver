/-- **First Baire theorem**: a completely metrizable topological space has Baire property.

Since `Mathlib` does not have the notion of a completely metrizable topological space yet,
we state it for a complete uniform space with countably generated uniformity filter. -/
instance (priority := 100) BaireSpace.of_pseudoEMetricSpace_completeSpace : BaireSpace X := by
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    ⊢ BaireSpace X
  -/
  let _ := UniformSpace.pseudoMetricSpace X
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    ⊢ BaireSpace X
  -/
  refine ⟨fun f ho hd => ?_⟩
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    ⊢ Dense (Set.iInter fun n => f n)
  -/
  let B : ℕ → ℝ≥0∞ := fun n => 1 / 2 ^ n
  have Bpos : ∀ n, 0 < B n := fun n ↦
    ENNReal.div_pos one_ne_zero <| ENNReal.pow_ne_top ENNReal.coe_ne_top
  /- Translate the density assumption into two functions `center` and `radius` associating
    to any n, x, δ, δpos a center and a positive radius such that
    `closedBall center radius` is included both in `f n` and in `closedBall x δ`.
    We can also require `radius ≤ (1/2)^(n+1)`, to ensure we get a Cauchy sequence later. -/
  have : ∀ n x δ, δ ≠ 0 → ∃ y r, 0 < r ∧ r ≤ B (n + 1) ∧ closedBall y r ⊆ closedBall x δ ∩ f n := by
    intro n x δ δpos
    have : x ∈ closure (f n) := hd n x
    rcases EMetric.mem_closure_iff.1 this (δ / 2) (ENNReal.half_pos δpos) with ⟨y, ys, xy⟩
    rw [edist_comm] at xy
    obtain ⟨r, rpos, hr⟩ : ∃ r > 0, closedBall y r ⊆ f n :=
      nhds_basis_closed_eball.mem_iff.1 (isOpen_iff_mem_nhds.1 (ho n) y ys)
    refine ⟨y, min (min (δ / 2) r) (B (n + 1)), ?_, ?_, fun z hz => ⟨?_, ?_⟩⟩
    · show 0 < min (min (δ / 2) r) (B (n + 1))
      exact lt_min (lt_min (ENNReal.half_pos δpos) rpos) (Bpos (n + 1))
    · show min (min (δ / 2) r) (B (n + 1)) ≤ B (n + 1)
      exact min_le_right _ _
    · show z ∈ closedBall x δ
      calc
        edist z x ≤ edist z y + edist y x := edist_triangle _ _ _
        _ ≤ min (min (δ / 2) r) (B (n + 1)) + δ / 2 := add_le_add hz (le_of_lt xy)
        _ ≤ δ / 2 + δ / 2 := (add_le_add (le_trans (min_le_left _ _) (min_le_left _ _)) le_rfl)
        _ = δ := ENNReal.add_halves δ
    show z ∈ f n
    exact hr (calc
      edist z y ≤ min (min (δ / 2) r) (B (n + 1)) := hz
      _ ≤ r := le_trans (min_le_left _ _) (min_le_right _ _))
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    this : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → Exists fun y => Exists fun  …
    ⊢ Dense (Set.iInter fun n => f n)
  -/
  choose! center radius Hpos HB Hball using this
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    ⊢ Dense (Set.iInter fun n => f n)
  -/
  refine fun x => (mem_closure_iff_nhds_basis nhds_basis_closed_eball).2 fun ε εpos => ?_
  /- `ε` is positive. We have to find a point in the ball of radius `ε` around `x` belonging to all
    `f n`. For this, we construct inductively a sequence `F n = (c n, r n)` such that the closed
    ball `closedBall (c n) (r n)` is included in the previous ball and in `f n`, and such that
    `r n` is small enough to ensure that `c n` is a Cauchy sequence. Then `c n` converges to a
    limit which belongs to all the `f n`. -/
  let F : ℕ → X × ℝ≥0∞ := fun n =>
    Nat.recOn n (Prod.mk x (min ε (B 0))) fun n p => Prod.mk (center n p.1 p.2) (radius n p.1 p.2)
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  let c : ℕ → X := fun n => (F n).1
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  let r : ℕ → ℝ≥0∞ := fun n => (F n).2
  have rpos : ∀ n, 0 < r n := by
    intro n
    induction n with
    | zero => exact lt_min εpos (Bpos 0)
    | succ n hn => exact Hpos n (c n) (r n) hn.ne'
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  have r0 : ∀ n, r n ≠ 0 := fun n => (rpos n).ne'
  have rB : ∀ n, r n ≤ B n := by
    intro n
    cases n with
    | zero => exact min_le_right _ _
    | succ n => exact HB n (c n) (r n) (r0 n)
  have incl : ∀ n, closedBall (c (n + 1)) (r (n + 1)) ⊆ closedBall (c n) (r n) ∩ f n :=
    fun n => Hball n (c n) (r n) (r0 n)
  have cdist : ∀ n, edist (c n) (c (n + 1)) ≤ B n := by
    intro n
    rw [edist_comm]
    have A : c (n + 1) ∈ closedBall (c (n + 1)) (r (n + 1)) := mem_closedBall_self
    have I :=
      calc
        closedBall (c (n + 1)) (r (n + 1)) ⊆ closedBall (c n) (r n) :=
          Subset.trans (incl n) inter_subset_left
        _ ⊆ closedBall (c n) (B n) := closedBall_subset_closedBall (rB n)
    exact I A
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  have : CauchySeq c := cauchySeq_of_edist_le_geometric_two _ ENNReal.one_ne_top cdist
  -- as the sequence `c n` is Cauchy in a complete space, it converges to a limit `y`.
  /-
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  rcases cauchySeq_tendsto_of_complete this with ⟨y, ylim⟩
  -- this point `y` will be the desired point. We will check that it belongs to all
  -- `f n` and to `ball x ε`.
  /-
    case intro
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    y : X
    ylim : Filter.Tendsto c Filter.atTop (nhds y)
    ⊢ Exists fun y => And (Membership.mem (Set.iInter fun n => f n) y) (Membership …
  -/
  use y
  /-
    case h
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    y : X
    ylim : Filter.Tendsto c Filter.atTop (nhds y)
    ⊢ And (Membership.mem (Set.iInter fun n => f n) y) (Membership.mem (EMetric.cl …
  -/
  simp only [exists_prop, Set.mem_iInter]
  have I : ∀ n, ∀ m ≥ n, closedBall (c m) (r m) ⊆ closedBall (c n) (r n) := by
    intro n
    refine Nat.le_induction ?_ fun m _ h => ?_
    · exact Subset.refl _
    · exact Subset.trans (incl m) (Subset.trans inter_subset_left h)
  have yball : ∀ n, y ∈ closedBall (c n) (r n) := by
    intro n
    refine isClosed_ball.mem_of_tendsto ylim ?_
    refine (Filter.eventually_ge_atTop n).mono fun m hm => ?_
    exact I n m hm mem_closedBall_self
  /-
    case h
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    y : X
    ylim : Filter.Tendsto c Filter.atTop (nhds y)
    I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
    yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
    ⊢ And (∀ (i : Nat), Membership.mem (f i) y) (Membership.mem (EMetric.closedBal …
  -/
  constructor
    /-
      case h.left
      X : Type u_1
      inst✝² : UniformSpace X
      inst✝¹ : CompleteSpace X
      inst✝ : (uniformity X).IsCountablyGenerated
      x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
      f : Nat → Set X
      ho : ∀ (n : Nat), IsOpen (f n)
      hd : ∀ (n : Nat), Dense (f n)
      B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
      Bpos : ∀ (n : Nat), LT.lt 0 (B n)
      center : Nat → X → ENNReal → X
      radius : Nat → X → ENNReal → ENNReal
      Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
      HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
      Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
      x : X
      ε : ENNReal
      εpos : LT.lt 0 ε
      F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
      c : Nat → X := fun n => (F n).1
      r : Nat → ENNReal := fun n => (F n).2
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      r0 : ∀ (n : Nat), Ne (r n) 0
      rB : ∀ (n : Nat), LE.le (r n) (B n)
      incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
      cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
      this : CauchySeq c
      y : X
      ylim : Filter.Tendsto c Filter.atTop (nhds y)
      I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
      yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
      ⊢ ∀ (i : Nat), Membership.mem (f i) y
    -/
  · show ∀ n, y ∈ f n
    /-
      case h.left
      X : Type u_1
      inst✝² : UniformSpace X
      inst✝¹ : CompleteSpace X
      inst✝ : (uniformity X).IsCountablyGenerated
      x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
      f : Nat → Set X
      ho : ∀ (n : Nat), IsOpen (f n)
      hd : ∀ (n : Nat), Dense (f n)
      B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
      Bpos : ∀ (n : Nat), LT.lt 0 (B n)
      center : Nat → X → ENNReal → X
      radius : Nat → X → ENNReal → ENNReal
      Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
      HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
      Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
      x : X
      ε : ENNReal
      εpos : LT.lt 0 ε
      F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
      c : Nat → X := fun n => (F n).1
      r : Nat → ENNReal := fun n => (F n).2
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      r0 : ∀ (n : Nat), Ne (r n) 0
      rB : ∀ (n : Nat), LE.le (r n) (B n)
      incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
      cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
      this : CauchySeq c
      y : X
      ylim : Filter.Tendsto c Filter.atTop (nhds y)
      I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
      yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
      ⊢ ∀ (n : Nat), Membership.mem (f n) y
    -/
    intro n
    have : closedBall (c (n + 1)) (r (n + 1)) ⊆ f n :=
      Subset.trans (incl n) inter_subset_right
    /-
      case h.left
      X : Type u_1
      inst✝² : UniformSpace X
      inst✝¹ : CompleteSpace X
      inst✝ : (uniformity X).IsCountablyGenerated
      x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
      f : Nat → Set X
      ho : ∀ (n : Nat), IsOpen (f n)
      hd : ∀ (n : Nat), Dense (f n)
      B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
      Bpos : ∀ (n : Nat), LT.lt 0 (B n)
      center : Nat → X → ENNReal → X
      radius : Nat → X → ENNReal → ENNReal
      Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
      HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
      Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
      x : X
      ε : ENNReal
      εpos : LT.lt 0 ε
      F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
      c : Nat → X := fun n => (F n).1
      r : Nat → ENNReal := fun n => (F n).2
      rpos : ∀ (n : Nat), LT.lt 0 (r n)
      r0 : ∀ (n : Nat), Ne (r n) 0
      rB : ∀ (n : Nat), LE.le (r n) (B n)
      incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
      cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
      this✝ : CauchySeq c
      y : X
      ylim : Filter.Tendsto c Filter.atTop (nhds y)
      I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
      yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
      n : Nat
      this : HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) (r (HAdd.hAdd  …
      ⊢ Membership.mem (f n) y
    -/
    exact this (yball (n + 1))
    /-
      🎉 no goals
    -/
  /-
    case h.right
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    y : X
    ylim : Filter.Tendsto c Filter.atTop (nhds y)
    I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
    yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
    ⊢ Membership.mem (EMetric.closedBall x ε) y
  -/
  show edist y x ≤ ε
  /-
    case h.right
    X : Type u_1
    inst✝² : UniformSpace X
    inst✝¹ : CompleteSpace X
    inst✝ : (uniformity X).IsCountablyGenerated
    x✝ : PseudoMetricSpace X := UniformSpace.pseudoMetricSpace X
    f : Nat → Set X
    ho : ∀ (n : Nat), IsOpen (f n)
    hd : ∀ (n : Nat), Dense (f n)
    B : Nat → ENNReal := fun n => HDiv.hDiv 1 (HPow.hPow 2 n)
    Bpos : ∀ (n : Nat), LT.lt 0 (B n)
    center : Nat → X → ENNReal → X
    radius : Nat → X → ENNReal → ENNReal
    Hpos : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LT.lt 0 (radius n x δ)
    HB : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → LE.le (radius n x δ) (B (HAdd …
    Hball : ∀ (n : Nat) (x : X) (δ : ENNReal), Ne δ 0 → HasSubset.Subset (EMetric. …
    x : X
    ε : ENNReal
    εpos : LT.lt 0 ε
    F : Nat → Prod X ENNReal := fun n => Nat.recOn n { fst := x, snd := Min.min ε  …
    c : Nat → X := fun n => (F n).1
    r : Nat → ENNReal := fun n => (F n).2
    rpos : ∀ (n : Nat), LT.lt 0 (r n)
    r0 : ∀ (n : Nat), Ne (r n) 0
    rB : ∀ (n : Nat), LE.le (r n) (B n)
    incl : ∀ (n : Nat), HasSubset.Subset (EMetric.closedBall (c (HAdd.hAdd n 1)) ( …
    cdist : ∀ (n : Nat), LE.le (EDist.edist (c n) (c (HAdd.hAdd n 1))) (B n)
    this : CauchySeq c
    y : X
    ylim : Filter.Tendsto c Filter.atTop (nhds y)
    I : ∀ (n m : Nat), GE.ge m n → HasSubset.Subset (EMetric.closedBall (c m) (r m …
    yball : ∀ (n : Nat), Membership.mem (EMetric.closedBall (c n) (r n)) y
    ⊢ LE.le (EDist.edist y x) ε
  -/
  exact le_trans (yball 0) (min_le_left _ _)
  /-
    🎉 no goals
  -/

