local infixr:25 " →ₛ " => SimpleFunc


/-- `nearestPtInd e N x` is the index `k` such that `e k` is the nearest point to `x` among the
points `e 0`, ..., `e N`. If more than one point are at the same distance from `x`, then
`nearestPtInd e N x` returns the least of their indexes. -/
noncomputable def nearestPtInd (e : ℕ → α) : ℕ → α →ₛ ℕ
  | 0 => const α 0
  | N + 1 =>
    piecewise (⋂ k ≤ N, { x | edist (e (N + 1)) x < edist (e k) x })
      (MeasurableSet.iInter fun _ =>
        MeasurableSet.iInter fun _ =>
          measurableSet_lt measurable_edist_right measurable_edist_right)
      (const α <| N + 1) (nearestPtInd e N)


/-- `nearestPt e N x` is the nearest point to `x` among the points `e 0`, ..., `e N`. If more than
one point are at the same distance from `x`, then `nearestPt e N x` returns the point with the
least possible index. -/
noncomputable def nearestPt (e : ℕ → α) (N : ℕ) : α →ₛ α :=
  (nearestPtInd e N).map e


@[simp]
theorem nearestPtInd_zero (e : ℕ → α) : nearestPtInd e 0 = const α 0 :=
  rfl


@[simp]
theorem nearestPt_zero (e : ℕ → α) : nearestPt e 0 = const α (e 0) :=
  rfl


theorem nearestPtInd_succ (e : ℕ → α) (N : ℕ) (x : α) :
    nearestPtInd e (N + 1) x =
      if ∀ k ≤ N, edist (e (N + 1)) x < edist (e k) x then N + 1 else nearestPtInd e N x := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    N : Nat
    x : α
    ⊢ Eq ((MeasureTheory.SimpleFunc.nearestPtInd e (HAdd.hAdd N 1)) x) (ite (∀ (k  …
  -/
  simp only [nearestPtInd, coe_piecewise, Set.piecewise]
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    N : Nat
    x : α
    ⊢ Eq (ite (Membership.mem (Set.iInter fun k => Set.iInter fun h => setOf fun x …
  -/
  congr
  /-
    case e_c
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    N : Nat
    x : α
    ⊢ Eq (Membership.mem (Set.iInter fun k => Set.iInter fun h => setOf fun x => L …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem nearestPtInd_le (e : ℕ → α) (N : ℕ) (x : α) : nearestPtInd e N x ≤ N := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    N : Nat
    x : α
    ⊢ LE.le ((MeasureTheory.SimpleFunc.nearestPtInd e N) x) N
  -/
  induction' N with N ihN; · simp
                             /-
                               🎉 no goals
                             -/
  /-
    case succ
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    N : Nat
    ihN : LE.le ((MeasureTheory.SimpleFunc.nearestPtInd e N) x) N
    ⊢ LE.le ((MeasureTheory.SimpleFunc.nearestPtInd e (HAdd.hAdd N 1)) x) (HAdd.hA …
  -/
  simp only [nearestPtInd_succ]
  /-
    case succ
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    N : Nat
    ihN : LE.le ((MeasureTheory.SimpleFunc.nearestPtInd e N) x) N
    ⊢ LE.le (ite (∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e (HAdd.hAdd N 1))  …
  -/
  split_ifs
  /-
    case pos
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    N : Nat
    ihN : LE.le ((MeasureTheory.SimpleFunc.nearestPtInd e N) x) N
    h✝ : ∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e (HAdd.hAdd N 1)) x) (EDist …
    ⊢ LE.le (HAdd.hAdd N 1) (HAdd.hAdd N 1)
  -/
  exacts [le_rfl, ihN.trans N.le_succ]
  /-
    🎉 no goals
  -/


theorem edist_nearestPt_le (e : ℕ → α) (x : α) {k N : ℕ} (hk : k ≤ N) :
    edist (nearestPt e N x) x ≤ edist (e k) x := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    k N : Nat
    hk : LE.le k N
    ⊢ LE.le (EDist.edist ((MeasureTheory.SimpleFunc.nearestPt e N) x) x) (EDist.ed …
  -/
  induction' N with N ihN generalizing k
    /-
      case zero
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : OpensMeasurableSpace α
      e : Nat → α
      x : α
      k : Nat
      hk : LE.le k 0
      ⊢ LE.le (EDist.edist ((MeasureTheory.SimpleFunc.nearestPt e 0) x) x) (EDist.ed …
    -/
  · simp [nonpos_iff_eq_zero.1 hk, le_refl]
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : OpensMeasurableSpace α
      e : Nat → α
      x : α
      N : Nat
      ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
      k : Nat
      hk : LE.le k (HAdd.hAdd N 1)
      ⊢ LE.le (EDist.edist ((MeasureTheory.SimpleFunc.nearestPt e (HAdd.hAdd N 1)) x …
    -/
  · simp only [nearestPt, nearestPtInd_succ, map_apply]
    /-
      case succ
      α : Type u_1
      inst✝² : MeasurableSpace α
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : OpensMeasurableSpace α
      e : Nat → α
      x : α
      N : Nat
      ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
      k : Nat
      hk : LE.le k (HAdd.hAdd N 1)
      ⊢ LE.le (EDist.edist (e (ite (∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e ( …
    -/
    split_ifs with h
      /-
        case pos
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        k : Nat
        hk : LE.le k (HAdd.hAdd N 1)
        h : ∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e (HAdd.hAdd N 1)) x) (EDist. …
        ⊢ LE.le (EDist.edist (e (HAdd.hAdd N 1)) x) (EDist.edist (e k) x)
      -/
    · rcases hk.eq_or_lt with (rfl | hk)
      /-
        case pos.inl
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        h : ∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e (HAdd.hAdd N 1)) x) (EDist. …
        hk : LE.le (HAdd.hAdd N 1) (HAdd.hAdd N 1)
        ⊢ LE.le (EDist.edist (e (HAdd.hAdd N 1)) x) (EDist.edist (e (HAdd.hAdd N 1)) x)
      -/
      exacts [le_rfl, (h k (Nat.lt_succ_iff.1 hk)).le]
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        k : Nat
        hk : LE.le k (HAdd.hAdd N 1)
        h : Not (∀ (k : Nat), LE.le k N → LT.lt (EDist.edist (e (HAdd.hAdd N 1)) x) (E …
        ⊢ LE.le (EDist.edist (e ((MeasureTheory.SimpleFunc.nearestPtInd e N) x)) x) (E …
      -/
    · push_neg at h
      /-
        case neg
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        k : Nat
        hk : LE.le k (HAdd.hAdd N 1)
        h : Exists fun k => And (LE.le k N) (LE.le (EDist.edist (e k) x) (EDist.edist  …
        ⊢ LE.le (EDist.edist (e ((MeasureTheory.SimpleFunc.nearestPtInd e N) x)) x) (E …
      -/
      rcases h with ⟨l, hlN, hxl⟩
      /-
        case neg.intro.intro
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        k : Nat
        hk : LE.le k (HAdd.hAdd N 1)
        l : Nat
        hlN : LE.le l N
        hxl : LE.le (EDist.edist (e l) x) (EDist.edist (e (HAdd.hAdd N 1)) x)
        ⊢ LE.le (EDist.edist (e ((MeasureTheory.SimpleFunc.nearestPtInd e N) x)) x) (E …
      -/
      rcases hk.eq_or_lt with (rfl | hk)
      /-
        case neg.intro.intro.inl
        α : Type u_1
        inst✝² : MeasurableSpace α
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : OpensMeasurableSpace α
        e : Nat → α
        x : α
        N : Nat
        ihN : ∀ {k : Nat}, LE.le k N → LE.le (EDist.edist ((MeasureTheory.SimpleFunc.n …
        l : Nat
        hlN : LE.le l N
        hxl : LE.le (EDist.edist (e l) x) (EDist.edist (e (HAdd.hAdd N 1)) x)
        hk : LE.le (HAdd.hAdd N 1) (HAdd.hAdd N 1)
        ⊢ LE.le (EDist.edist (e ((MeasureTheory.SimpleFunc.nearestPtInd e N) x)) x) (E …
      -/
      exacts [(ihN hlN).trans hxl, ihN (Nat.lt_succ_iff.1 hk)]
      /-
        🎉 no goals
      -/


theorem tendsto_nearestPt {e : ℕ → α} {x : α} (hx : x ∈ closure (range e)) :
    Tendsto (fun N => nearestPt e N x) atTop (𝓝 x) := by
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    hx : Membership.mem (closure (Set.range e)) x
    ⊢ Filter.Tendsto (fun N => (MeasureTheory.SimpleFunc.nearestPt e N) x) Filter. …
  -/
  refine (atTop_basis.tendsto_iff nhds_basis_eball).2 fun ε hε => ?_
  /-
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    hx : Membership.mem (closure (Set.range e)) x
    ε : ENNReal
    hε : LT.lt 0 ε
    ⊢ Exists fun ia => And True (∀ (x_1 : Nat), Membership.mem (Set.Ici ia) x_1 →  …
  -/
  rcases EMetric.mem_closure_iff.1 hx ε hε with ⟨_, ⟨N, rfl⟩, hN⟩
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    hx : Membership.mem (closure (Set.range e)) x
    ε : ENNReal
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (EDist.edist x (e N)) ε
    ⊢ Exists fun ia => And True (∀ (x_1 : Nat), Membership.mem (Set.Ici ia) x_1 →  …
  -/
  rw [edist_comm] at hN
  /-
    case intro.intro.intro
    α : Type u_1
    inst✝² : MeasurableSpace α
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : OpensMeasurableSpace α
    e : Nat → α
    x : α
    hx : Membership.mem (closure (Set.range e)) x
    ε : ENNReal
    hε : LT.lt 0 ε
    N : Nat
    hN : LT.lt (EDist.edist (e N) x) ε
    ⊢ Exists fun ia => And True (∀ (x_1 : Nat), Membership.mem (Set.Ici ia) x_1 →  …
  -/
  exact ⟨N, trivial, fun n hn => (edist_nearestPt_le e x hn).trans_lt hN⟩
  /-
    🎉 no goals
  -/


/-- Approximate a measurable function by a sequence of simple functions `F n` such that
`F n x ∈ s`. -/
noncomputable def approxOn (f : β → α) (hf : Measurable f) (s : Set α) (y₀ : α) (h₀ : y₀ ∈ s)
    [SeparableSpace s] (n : ℕ) : β →ₛ α :=
  haveI : Nonempty s := ⟨⟨y₀, h₀⟩⟩
  comp (nearestPt (fun k => Nat.casesOn k y₀ ((↑) ∘ denseSeq s) : ℕ → α) n) f hf


@[simp]
theorem approxOn_zero {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] (x : β) : approxOn f hf s y₀ h₀ 0 x = y₀ :=
  rfl


theorem approxOn_mem {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] (n : ℕ) (x : β) : approxOn f hf s y₀ h₀ n x ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    n : Nat
    x : β
    ⊢ Membership.mem s ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n) x)
  -/
  haveI : Nonempty s := ⟨⟨y₀, h₀⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    n : Nat
    x : β
    this : Nonempty ↑s
    ⊢ Membership.mem s ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n) x)
  -/
  suffices ∀ n, (Nat.casesOn n y₀ ((↑) ∘ denseSeq s) : α) ∈ s by apply this
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    n : Nat
    x : β
    this : Nonempty ↑s
    ⊢ ∀ (n : Nat), Membership.mem s (Nat.casesOn n y₀ (Function.comp Subtype.val ( …
  -/
  rintro (_ | n)
  /-
    case zero
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    n : Nat
    x : β
    this : Nonempty ↑s
    ⊢ Membership.mem s (Nat.casesOn 0 y₀ (Function.comp Subtype.val (TopologicalSp …
  -/
  exacts [h₀, Subtype.mem _]
  /-
    🎉 no goals
  -/


@[simp, nolint simpNF] -- Porting note: LHS doesn't simplify.
theorem approxOn_comp {γ : Type*} [MeasurableSpace γ] {f : β → α} (hf : Measurable f) {g : γ → β}
    (hg : Measurable g) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s) [SeparableSpace s] (n : ℕ) :
    approxOn (f ∘ g) (hf.comp hg) s y₀ h₀ n = (approxOn f hf s y₀ h₀ n).comp g hg :=
  rfl


theorem tendsto_approxOn {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] {x : β} (hx : f x ∈ closure s) :
    Tendsto (fun n => approxOn f hf s y₀ h₀ n x) atTop (𝓝 <| f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    hx : Membership.mem (closure s) (f x)
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n)  …
  -/
  haveI : Nonempty s := ⟨⟨y₀, h₀⟩⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    hx : Membership.mem (closure s) (f x)
    this : Nonempty ↑s
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n)  …
  -/
  rw [← @Subtype.range_coe _ s, ← image_univ, ← (denseRange_denseSeq s).closure_eq] at hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    this : Nonempty ↑s
    hx : Membership.mem (closure (Set.image Subtype.val (closure (Set.range (Topol …
    ⊢ Filter.Tendsto (fun n => (MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n)  …
  -/
  simp (config := { iota := false }) only [approxOn, coe_comp]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    this : Nonempty ↑s
    hx : Membership.mem (closure (Set.image Subtype.val (closure (Set.range (Topol …
    ⊢ Filter.Tendsto (fun n => Function.comp (⇑(MeasureTheory.SimpleFunc.nearestPt …
  -/
  refine tendsto_nearestPt (closure_minimal ?_ isClosed_closure hx)
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    this : Nonempty ↑s
    hx : Membership.mem (closure (Set.image Subtype.val (closure (Set.range (Topol …
    ⊢ HasSubset.Subset (Set.image Subtype.val (closure (Set.range (TopologicalSpac …
  -/
  simp (config := { iota := false }) only [Nat.range_casesOn, closure_union, range_comp]
  exact
    Subset.trans (image_closure_subset_closure_image continuous_subtype_val)
      subset_union_right


theorem edist_approxOn_mono {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] (x : β) {m n : ℕ} (h : m ≤ n) :
    edist (approxOn f hf s y₀ h₀ n x) (f x) ≤ edist (approxOn f hf s y₀ h₀ m x) (f x) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    m n : Nat
    h : LE.le m n
    ⊢ LE.le (EDist.edist ((MeasureTheory.SimpleFunc.approxOn f hf s y₀ h₀ n) x) (f …
  -/
  dsimp only [approxOn, coe_comp, Function.comp_def]
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁴ : MeasurableSpace α
    inst✝³ : PseudoEMetricSpace α
    inst✝² : OpensMeasurableSpace α
    inst✝¹ : MeasurableSpace β
    f : β → α
    hf : Measurable f
    s : Set α
    y₀ : α
    h₀ : Membership.mem s y₀
    inst✝ : TopologicalSpace.SeparableSpace ↑s
    x : β
    m n : Nat
    h : LE.le m n
    ⊢ LE.le (EDist.edist ((MeasureTheory.SimpleFunc.nearestPt (fun k => Nat.rec y₀ …
  -/
  exact edist_nearestPt_le _ _ ((nearestPtInd_le _ _ _).trans h)
  /-
    🎉 no goals
  -/


theorem edist_approxOn_le {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] (x : β) (n : ℕ) : edist (approxOn f hf s y₀ h₀ n x) (f x) ≤ edist y₀ (f x) :=
  edist_approxOn_mono hf h₀ x (zero_le n)


theorem edist_approxOn_y0_le {f : β → α} (hf : Measurable f) {s : Set α} {y₀ : α} (h₀ : y₀ ∈ s)
    [SeparableSpace s] (x : β) (n : ℕ) :
    edist y₀ (approxOn f hf s y₀ h₀ n x) ≤ edist y₀ (f x) + edist y₀ (f x) :=
  calc
    edist y₀ (approxOn f hf s y₀ h₀ n x) ≤
        edist y₀ (f x) + edist (approxOn f hf s y₀ h₀ n x) (f x) :=
      edist_triangle_right _ _ _
    _ ≤ edist y₀ (f x) + edist y₀ (f x) := add_le_add_left (edist_approxOn_le hf h₀ x n) _


/-- A continuous function with compact support on a product space can be uniformly approximated by
simple functions. The subtlety is that we do not assume that the spaces are separable, so the
product of the Borel sigma algebras might not contain all open sets, but still it contains enough
of them to approximate compactly supported continuous functions. -/
lemma HasCompactSupport.exists_simpleFunc_approx_of_prod [PseudoMetricSpace α]
    {f : X × Y → α} (hf : Continuous f) (h'f : HasCompactSupport f)
    {ε : ℝ} (hε : 0 < ε) :
    ∃ (g : SimpleFunc (X × Y) α), ∀ x, dist (f x) (g x) < ε := by
  have M : ∀ (K : Set (X × Y)), IsCompact K →
      ∃ (g : SimpleFunc (X × Y) α), ∃ (s : Set (X × Y)), MeasurableSet s ∧ K ⊆ s ∧
      ∀ x ∈ s, dist (f x) (g x) < ε := by
    intro K hK
    apply IsCompact.induction_on
      (p := fun t ↦ ∃ (g : SimpleFunc (X × Y) α), ∃ (s : Set (X × Y)), MeasurableSet s ∧ t ⊆ s ∧
        ∀ x ∈ s, dist (f x) (g x) < ε) hK
    · exact ⟨0, ∅, by simp⟩
    · intro t t' htt' ⟨g, s, s_meas, ts, hg⟩
      exact ⟨g, s, s_meas, htt'.trans ts, hg⟩
    · intro t t' ⟨g, s, s_meas, ts, hg⟩ ⟨g', s', s'_meas, t's', hg'⟩
      refine ⟨g.piecewise s s_meas g', s ∪ s', s_meas.union s'_meas,
        union_subset_union ts t's', fun p hp ↦ ?_⟩
      by_cases H : p ∈ s
      · simpa [H, SimpleFunc.piecewise_apply] using hg p H
      · simp only [SimpleFunc.piecewise_apply, H, ite_false]
        apply hg'
        simpa [H] using (mem_union _ _ _).1 hp
    · rintro ⟨x, y⟩ -
      obtain ⟨u, v, hu, xu, hv, yv, huv⟩ : ∃ u v, IsOpen u ∧ x ∈ u ∧ IsOpen v ∧ y ∈ v ∧
        u ×ˢ v ⊆ {z | dist (f z) (f (x, y)) < ε} :=
          mem_nhds_prod_iff'.1 <| Metric.continuousAt_iff'.1 hf.continuousAt ε hε
      refine ⟨u ×ˢ v, nhdsWithin_le_nhds <| (hu.prod hv).mem_nhds (mk_mem_prod xu yv), ?_⟩
      exact ⟨SimpleFunc.const _ (f (x, y)), u ×ˢ v, hu.measurableSet.prod hv.measurableSet,
        Subset.rfl, fun z hz ↦ huv hz⟩
  obtain ⟨g, s, s_meas, fs, hg⟩ : ∃ (g : SimpleFunc (X × Y) α) (s : Set (X × Y)),
    MeasurableSet s ∧ tsupport f ⊆ s ∧ ∀ (x : X × Y), x ∈ s → dist (f x) (g x) < ε := M _ h'f
  /-
    case intro.intro.intro.intro
    X : Type u_7
    Y : Type u_8
    α : Type u_9
    inst✝⁷ : Zero α
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace X
    inst✝³ : MeasurableSpace Y
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : OpensMeasurableSpace Y
    inst✝ : PseudoMetricSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    ε : Real
    hε : LT.lt 0 ε
    M : ∀ (K : Set (Prod X Y)), IsCompact K → Exists fun g => Exists fun s => And  …
    g : MeasureTheory.SimpleFunc (Prod X Y) α
    s : Set (Prod X Y)
    s_meas : MeasurableSet s
    fs : HasSubset.Subset (tsupport f) s
    hg : ∀ (x : Prod X Y), Membership.mem s x → LT.lt (Dist.dist (f x) (g x)) ε
    ⊢ Exists fun g => ∀ (x : Prod X Y), LT.lt (Dist.dist (f x) (g x)) ε
  -/
  refine ⟨g.piecewise s s_meas 0, fun p ↦ ?_⟩
  /-
    case intro.intro.intro.intro
    X : Type u_7
    Y : Type u_8
    α : Type u_9
    inst✝⁷ : Zero α
    inst✝⁶ : TopologicalSpace X
    inst✝⁵ : TopologicalSpace Y
    inst✝⁴ : MeasurableSpace X
    inst✝³ : MeasurableSpace Y
    inst✝² : OpensMeasurableSpace X
    inst✝¹ : OpensMeasurableSpace Y
    inst✝ : PseudoMetricSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    ε : Real
    hε : LT.lt 0 ε
    M : ∀ (K : Set (Prod X Y)), IsCompact K → Exists fun g => Exists fun s => And  …
    g : MeasureTheory.SimpleFunc (Prod X Y) α
    s : Set (Prod X Y)
    s_meas : MeasurableSet s
    fs : HasSubset.Subset (tsupport f) s
    hg : ∀ (x : Prod X Y), Membership.mem s x → LT.lt (Dist.dist (f x) (g x)) ε
    p : Prod X Y
    ⊢ LT.lt (Dist.dist (f p) ((MeasureTheory.SimpleFunc.piecewise s s_meas g 0) p) …
  -/
  by_cases H : p ∈ s
    /-
      case pos
      X : Type u_7
      Y : Type u_8
      α : Type u_9
      inst✝⁷ : Zero α
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace X
      inst✝³ : MeasurableSpace Y
      inst✝² : OpensMeasurableSpace X
      inst✝¹ : OpensMeasurableSpace Y
      inst✝ : PseudoMetricSpace α
      f : Prod X Y → α
      hf : Continuous f
      h'f : HasCompactSupport f
      ε : Real
      hε : LT.lt 0 ε
      M : ∀ (K : Set (Prod X Y)), IsCompact K → Exists fun g => Exists fun s => And  …
      g : MeasureTheory.SimpleFunc (Prod X Y) α
      s : Set (Prod X Y)
      s_meas : MeasurableSet s
      fs : HasSubset.Subset (tsupport f) s
      hg : ∀ (x : Prod X Y), Membership.mem s x → LT.lt (Dist.dist (f x) (g x)) ε
      p : Prod X Y
      H : Membership.mem s p
      ⊢ LT.lt (Dist.dist (f p) ((MeasureTheory.SimpleFunc.piecewise s s_meas g 0) p) …
    -/
  · simpa [H, SimpleFunc.piecewise_apply] using hg p H
    /-
      🎉 no goals
    -/
  · have : f p = 0 := by
      contrapose! H
      rw [← Function.mem_support] at H
      exact fs (subset_tsupport _ H)
    /-
      case neg
      X : Type u_7
      Y : Type u_8
      α : Type u_9
      inst✝⁷ : Zero α
      inst✝⁶ : TopologicalSpace X
      inst✝⁵ : TopologicalSpace Y
      inst✝⁴ : MeasurableSpace X
      inst✝³ : MeasurableSpace Y
      inst✝² : OpensMeasurableSpace X
      inst✝¹ : OpensMeasurableSpace Y
      inst✝ : PseudoMetricSpace α
      f : Prod X Y → α
      hf : Continuous f
      h'f : HasCompactSupport f
      ε : Real
      hε : LT.lt 0 ε
      M : ∀ (K : Set (Prod X Y)), IsCompact K → Exists fun g => Exists fun s => And  …
      g : MeasureTheory.SimpleFunc (Prod X Y) α
      s : Set (Prod X Y)
      s_meas : MeasurableSet s
      fs : HasSubset.Subset (tsupport f) s
      hg : ∀ (x : Prod X Y), Membership.mem s x → LT.lt (Dist.dist (f x) (g x)) ε
      p : Prod X Y
      H : Not (Membership.mem s p)
      this : Eq (f p) 0
      ⊢ LT.lt (Dist.dist (f p) ((MeasureTheory.SimpleFunc.piecewise s s_meas g 0) p) …
    -/
    simp [SimpleFunc.piecewise_apply, H, ite_false, this, hε]
    /-
      🎉 no goals
    -/


/-- A continuous function with compact support on a product space is measurable for the product
sigma-algebra. The subtlety is that we do not assume that the spaces are separable, so the
product of the Borel sigma algebras might not contain all open sets, but still it contains enough
of them to approximate compactly supported continuous functions. -/
lemma HasCompactSupport.measurable_of_prod
    [TopologicalSpace α] [PseudoMetrizableSpace α] [MeasurableSpace α] [BorelSpace α]
    {f : X × Y → α} (hf : Continuous f) (h'f : HasCompactSupport f) :
    Measurable f := by
  /-
    X : Type u_7
    Y : Type u_8
    α : Type u_9
    inst✝¹⁰ : Zero α
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : OpensMeasurableSpace Y
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace.PseudoMetrizableSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    ⊢ Measurable f
  -/
  letI : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetric α
  obtain ⟨u, -, u_pos, u_lim⟩ : ∃ u, StrictAnti u ∧ (∀ (n : ℕ), 0 < u n) ∧ Tendsto u atTop (𝓝 0) :=
    exists_seq_strictAnti_tendsto (0 : ℝ)
  have : ∀ n, ∃ (g : SimpleFunc (X × Y) α), ∀ x, dist (f x) (g x) < u n :=
    fun n ↦ h'f.exists_simpleFunc_approx_of_prod hf (u_pos n)
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    α : Type u_9
    inst✝¹⁰ : Zero α
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : OpensMeasurableSpace Y
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace.PseudoMetrizableSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    this✝ : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMet …
    u : Nat → Real
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    this : ∀ (n : Nat), Exists fun g => ∀ (x : Prod X Y), LT.lt (Dist.dist (f x) ( …
    ⊢ Measurable f
  -/
  choose g hg using this
  have A : ∀ x, Tendsto (fun n ↦ g n x) atTop (𝓝 (f x)) := by
    intro x
    rw [tendsto_iff_dist_tendsto_zero]
    apply squeeze_zero (fun n ↦ dist_nonneg) (fun n ↦ ?_) u_lim
    rw [dist_comm]
    exact (hg n x).le
  /-
    case intro.intro.intro
    X : Type u_7
    Y : Type u_8
    α : Type u_9
    inst✝¹⁰ : Zero α
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace X
    inst✝⁶ : MeasurableSpace Y
    inst✝⁵ : OpensMeasurableSpace X
    inst✝⁴ : OpensMeasurableSpace Y
    inst✝³ : TopologicalSpace α
    inst✝² : TopologicalSpace.PseudoMetrizableSpace α
    inst✝¹ : MeasurableSpace α
    inst✝ : BorelSpace α
    f : Prod X Y → α
    hf : Continuous f
    h'f : HasCompactSupport f
    this : PseudoMetricSpace α := TopologicalSpace.pseudoMetrizableSpacePseudoMetr …
    u : Nat → Real
    u_pos : ∀ (n : Nat), LT.lt 0 (u n)
    u_lim : Filter.Tendsto u Filter.atTop (nhds 0)
    g : Nat → MeasureTheory.SimpleFunc (Prod X Y) α
    hg : ∀ (n : Nat) (x : Prod X Y), LT.lt (Dist.dist (f x) ((g n) x)) (u n)
    A : ∀ (x : Prod X Y), Filter.Tendsto (fun n => (g n) x) Filter.atTop (nhds (f  …
    ⊢ Measurable f
  -/
  apply measurable_of_tendsto_metrizable (fun n ↦ (g n).measurable) (tendsto_pi_nhds.2 A)
  /-
    🎉 no goals
  -/


