/-- The distance on the completion is obtained by extending the distance on the original space,
by uniform continuity. -/
instance : Dist (Completion α) :=
  ⟨Completion.extension₂ dist⟩


/-- The new distance is uniformly continuous. -/
protected theorem uniformContinuous_dist :
    UniformContinuous fun p : Completion α × Completion α ↦ dist p.1 p.2 :=
  uniformContinuous_extension₂ dist


/-- The new distance is continuous. -/
protected theorem continuous_dist [TopologicalSpace β] {f g : β → Completion α} (hf : Continuous f)
    (hg : Continuous g) : Continuous fun x ↦ dist (f x) (g x) :=
  Completion.uniformContinuous_dist.continuous.comp (hf.prod_mk hg : _)


/-- The new distance is an extension of the original distance. -/
@[simp]
protected theorem dist_eq (x y : α) : dist (x : Completion α) y = dist x y :=
  Completion.extension₂_coe_coe uniformContinuous_dist _ _

/- Let us check that the new distance satisfies the axioms of a distance, by starting from the
properties on α and extending them to `Completion α` by continuity. -/

protected theorem dist_self (x : Completion α) : dist x x = 0 := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x : UniformSpace.Completion α
    ⊢ Eq (Dist.dist x x) 0
  -/
  refine induction_on x ?_ ?_
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoMetricSpace α
      x : UniformSpace.Completion α
      ⊢ IsClosed (setOf fun a => Eq (Dist.dist a a) 0)
    -/
  · refine isClosed_eq ?_ continuous_const
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoMetricSpace α
      x : UniformSpace.Completion α
      ⊢ Continuous fun a => Dist.dist a a
    -/
    exact Completion.continuous_dist continuous_id continuous_id
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x : UniformSpace.Completion α
      ⊢ ∀ (a : α), Eq (Dist.dist (↑α a) (↑α a)) 0
    -/
  · intro a
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x : UniformSpace.Completion α
      a : α
      ⊢ Eq (Dist.dist (↑α a) (↑α a)) 0
    -/
    rw [Completion.dist_eq, dist_self]
    /-
      🎉 no goals
    -/


protected theorem dist_comm (x y : Completion α) : dist x y = dist y x := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y : UniformSpace.Completion α
    ⊢ Eq (Dist.dist x y) (Dist.dist y x)
  -/
  refine induction_on₂ x y ?_ ?_
  · exact isClosed_eq (Completion.continuous_dist continuous_fst continuous_snd)
        (Completion.continuous_dist continuous_snd continuous_fst)
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y : UniformSpace.Completion α
      ⊢ ∀ (a b : α), Eq (Dist.dist (↑α a) (↑α b)) (Dist.dist (↑α b) (↑α a))
    -/
  · intro a b
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y : UniformSpace.Completion α
      a b : α
      ⊢ Eq (Dist.dist (↑α a) (↑α b)) (Dist.dist (↑α b) (↑α a))
    -/
    rw [Completion.dist_eq, Completion.dist_eq, dist_comm]
    /-
      🎉 no goals
    -/


protected theorem dist_triangle (x y z : Completion α) : dist x z ≤ dist x y + dist y z := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    x y z : UniformSpace.Completion α
    ⊢ LE.le (Dist.dist x z) (HAdd.hAdd (Dist.dist x y) (Dist.dist y z))
  -/
  refine induction_on₃ x y z ?_ ?_
    /-
      case refine_1
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y z : UniformSpace.Completion α
      ⊢ IsClosed (setOf fun x => LE.le (Dist.dist x.1 x.2.2) (HAdd.hAdd (Dist.dist x …
    -/
  · refine isClosed_le ?_ (Continuous.add ?_ ?_) <;>
      /-
        case refine_1.refine_1
        α : Type u
        inst✝ : PseudoMetricSpace α
        x y z : UniformSpace.Completion α
        ⊢ Continuous fun x => Dist.dist x.1 x.2.2
      -/
      /-
        🎉 no goals
      -/
      /-
        🎉 no goals
      -/
      apply_rules [Completion.continuous_dist, Continuous.fst, Continuous.snd, continuous_id]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y z : UniformSpace.Completion α
      ⊢ ∀ (a b c : α), LE.le (Dist.dist (↑α a) (↑α c)) (HAdd.hAdd (Dist.dist (↑α a)  …
    -/
  · intro a b c
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y z : UniformSpace.Completion α
      a b c : α
      ⊢ LE.le (Dist.dist (↑α a) (↑α c)) (HAdd.hAdd (Dist.dist (↑α a) (↑α b)) (Dist.d …
    -/
    rw [Completion.dist_eq, Completion.dist_eq, Completion.dist_eq]
    /-
      case refine_2
      α : Type u
      inst✝ : PseudoMetricSpace α
      x y z : UniformSpace.Completion α
      a b c : α
      ⊢ LE.le (Dist.dist a c) (HAdd.hAdd (Dist.dist a b) (Dist.dist b c))
    -/
    exact dist_triangle a b c
    /-
      🎉 no goals
    -/


/-- Elements of the uniformity (defined generally for completions) can be characterized in terms
of the distance. -/
protected theorem mem_uniformity_dist (s : Set (Completion α × Completion α)) :
    s ∈ 𝓤 (Completion α) ↔ ∃ ε > 0, ∀ {a b}, dist a b < ε → (a, b) ∈ s := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
    ⊢ Iff (Membership.mem (uniformity (UniformSpace.Completion α)) s) (Exists fun  …
  -/
  constructor
  · /- Start from an entourage `s`. It contains a closed entourage `t`. Its pullback in `α` is an
      entourage, so it contains an `ε`-neighborhood of the diagonal by definition of the entourages
      in metric spaces. Then `t` contains an `ε`-neighborhood of the diagonal in `Completion α`, as
      closed properties pass to the completion. -/
    /-
      case mp
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s → Exists fun ε =>  …
    -/
    intro hs
    /-
      case mp
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      hs : Membership.mem (uniformity (UniformSpace.Completion α)) s
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ {a b : UniformSpace.Completion α}, LT.lt  …
    -/
    rcases mem_uniformity_isClosed hs with ⟨t, ht, ⟨tclosed, ts⟩⟩
    have A : { x : α × α | (↑x.1, ↑x.2) ∈ t } ∈ uniformity α :=
      uniformContinuous_def.1 (uniformContinuous_coe α) t ht
    /-
      case mp.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      hs : Membership.mem (uniformity (UniformSpace.Completion α)) s
      t : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht : Membership.mem (uniformity (UniformSpace.Completion α)) t
      tclosed : IsClosed t
      ts : HasSubset.Subset t s
      A : Membership.mem (uniformity α) (setOf fun x => Membership.mem t { fst := ↑α …
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ {a b : UniformSpace.Completion α}, LT.lt  …
    -/
    rcases mem_uniformity_dist.1 A with ⟨ε, εpos, hε⟩
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      hs : Membership.mem (uniformity (UniformSpace.Completion α)) s
      t : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht : Membership.mem (uniformity (UniformSpace.Completion α)) t
      tclosed : IsClosed t
      ts : HasSubset.Subset t s
      A : Membership.mem (uniformity α) (setOf fun x => Membership.mem t { fst := ↑α …
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem (setOf fun x => Mem …
      ⊢ Exists fun ε => And (GT.gt ε 0) (∀ {a b : UniformSpace.Completion α}, LT.lt  …
    -/
    refine ⟨ε, εpos, @fun x y hxy ↦ ?_⟩
    have : ε ≤ dist x y ∨ (x, y) ∈ t := by
      refine induction_on₂ x y ?_ ?_
      · have : { x : Completion α × Completion α | ε ≤ dist x.fst x.snd ∨ (x.fst, x.snd) ∈ t } =
               { p : Completion α × Completion α | ε ≤ dist p.1 p.2 } ∪ t := by ext; simp
        rw [this]
        apply IsClosed.union _ tclosed
        exact isClosed_le continuous_const Completion.uniformContinuous_dist.continuous
      · intro x y
        rw [Completion.dist_eq]
        by_cases h : ε ≤ dist x y
        · exact Or.inl h
        · have Z := hε (not_le.1 h)
          simp only [Set.mem_setOf_eq] at Z
          exact Or.inr Z
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      hs : Membership.mem (uniformity (UniformSpace.Completion α)) s
      t : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht : Membership.mem (uniformity (UniformSpace.Completion α)) t
      tclosed : IsClosed t
      ts : HasSubset.Subset t s
      A : Membership.mem (uniformity α) (setOf fun x => Membership.mem t { fst := ↑α …
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem (setOf fun x => Mem …
      x y : UniformSpace.Completion α
      hxy : LT.lt (Dist.dist x y) ε
      this : Or (LE.le ε (Dist.dist x y)) (Membership.mem t { fst := x, snd := y })
      ⊢ Membership.mem s { fst := x, snd := y }
    -/
    simp only [not_le.mpr hxy, false_or, not_le] at this
    /-
      case mp.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      hs : Membership.mem (uniformity (UniformSpace.Completion α)) s
      t : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht : Membership.mem (uniformity (UniformSpace.Completion α)) t
      tclosed : IsClosed t
      ts : HasSubset.Subset t s
      A : Membership.mem (uniformity α) (setOf fun x => Membership.mem t { fst := ↑α …
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ ⦃a b : α⦄, LT.lt (Dist.dist a b) ε → Membership.mem (setOf fun x => Mem …
      x y : UniformSpace.Completion α
      hxy : LT.lt (Dist.dist x y) ε
      this : Membership.mem t { fst := x, snd := y }
      ⊢ Membership.mem s { fst := x, snd := y }
    -/
    exact ts this
    /-
      🎉 no goals
    -/
  · /- Start from a set `s` containing an ε-neighborhood of the diagonal in `Completion α`. To show
        that it is an entourage, we use the fact that `dist` is uniformly continuous on
        `Completion α × Completion α` (this is a general property of the extension of uniformly
        continuous functions). Therefore, the preimage of the ε-neighborhood of the diagonal in ℝ
        is an entourage in `Completion α × Completion α`. Massaging this property, it follows that
        the ε-neighborhood of the diagonal is an entourage in `Completion α`, and therefore this is
        also the case of `s`. -/
    /-
      case mpr
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ⊢ (Exists fun ε => And (GT.gt ε 0) (∀ {a b : UniformSpace.Completion α}, LT.lt …
    -/
    rintro ⟨ε, εpos, hε⟩
    /-
      case mpr.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s
    -/
    let r : Set (ℝ × ℝ) := { p | dist p.1 p.2 < ε }
    /-
      case mpr.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s
    -/
    have : r ∈ uniformity ℝ := Metric.dist_mem_uniformity εpos
    /-
      case mpr.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s
    -/
    have T := uniformContinuous_def.1 (@Completion.uniformContinuous_dist α _) r this
    simp only [uniformity_prod_eq_prod, mem_prod_iff, exists_prop, Filter.mem_map,
      Set.mem_setOf_eq] at T
    /-
      case mpr.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      T : Exists fun t₁ => And (Membership.mem (uniformity (UniformSpace.Completion  …
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s
    -/
    rcases T with ⟨t1, ht1, t2, ht2, ht⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      t1 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht1 : Membership.mem (uniformity (UniformSpace.Completion α)) t1
      t2 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht2 : Membership.mem (uniformity (UniformSpace.Completion α)) t2
      ht : HasSubset.Subset (SProd.sprod t1 t2) (Set.preimage (fun p => { fst := { f …
      ⊢ Membership.mem (uniformity (UniformSpace.Completion α)) s
    -/
    refine mem_of_superset ht1 ?_
    have A : ∀ a b : Completion α, (a, b) ∈ t1 → dist a b < ε := by
      intro a b hab
      have : ((a, b), (a, a)) ∈ t1 ×ˢ t2 := ⟨hab, refl_mem_uniformity ht2⟩
      have I := ht this
      simp? [r, Completion.dist_self, Real.dist_eq, Completion.dist_comm] at I says
        simp only [Real.dist_eq, mem_setOf_eq, preimage_setOf_eq, Completion.dist_self,
          Completion.dist_comm, zero_sub, abs_neg, r] at I
      exact lt_of_le_of_lt (le_abs_self _) I
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      t1 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht1 : Membership.mem (uniformity (UniformSpace.Completion α)) t1
      t2 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht2 : Membership.mem (uniformity (UniformSpace.Completion α)) t2
      ht : HasSubset.Subset (SProd.sprod t1 t2) (Set.preimage (fun p => { fst := { f …
      A : ∀ (a b : UniformSpace.Completion α), Membership.mem t1 { fst := a, snd :=  …
      ⊢ HasSubset.Subset t1 s
    -/
    show t1 ⊆ s
    /-
      case mpr.intro.intro.intro.intro.intro.intro
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      t1 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht1 : Membership.mem (uniformity (UniformSpace.Completion α)) t1
      t2 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht2 : Membership.mem (uniformity (UniformSpace.Completion α)) t2
      ht : HasSubset.Subset (SProd.sprod t1 t2) (Set.preimage (fun p => { fst := { f …
      A : ∀ (a b : UniformSpace.Completion α), Membership.mem t1 { fst := a, snd :=  …
      ⊢ HasSubset.Subset t1 s
    -/
    rintro ⟨a, b⟩ hp
    /-
      case mpr.intro.intro.intro.intro.intro.intro.mk
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this : Membership.mem (uniformity Real) r
      t1 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht1 : Membership.mem (uniformity (UniformSpace.Completion α)) t1
      t2 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht2 : Membership.mem (uniformity (UniformSpace.Completion α)) t2
      ht : HasSubset.Subset (SProd.sprod t1 t2) (Set.preimage (fun p => { fst := { f …
      A : ∀ (a b : UniformSpace.Completion α), Membership.mem t1 { fst := a, snd :=  …
      a b : UniformSpace.Completion α
      hp : Membership.mem t1 { fst := a, snd := b }
      ⊢ Membership.mem s { fst := a, snd := b }
    -/
    have : dist a b < ε := A a b hp
    /-
      case mpr.intro.intro.intro.intro.intro.intro.mk
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ε : Real
      εpos : GT.gt ε 0
      hε : ∀ {a b : UniformSpace.Completion α}, LT.lt (Dist.dist a b) ε → Membership …
      r : Set (Prod Real Real) := setOf fun p => LT.lt (Dist.dist p.1 p.2) ε
      this✝ : Membership.mem (uniformity Real) r
      t1 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht1 : Membership.mem (uniformity (UniformSpace.Completion α)) t1
      t2 : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ht2 : Membership.mem (uniformity (UniformSpace.Completion α)) t2
      ht : HasSubset.Subset (SProd.sprod t1 t2) (Set.preimage (fun p => { fst := { f …
      A : ∀ (a b : UniformSpace.Completion α), Membership.mem t1 { fst := a, snd :=  …
      a b : UniformSpace.Completion α
      hp : Membership.mem t1 { fst := a, snd := b }
      this : LT.lt (Dist.dist a b) ε
      ⊢ Membership.mem s { fst := a, snd := b }
    -/
    exact hε this
    /-
      🎉 no goals
    -/


/-- Reformulate `Completion.mem_uniformity_dist` in terms that are suitable for the definition
of the metric space structure. -/
protected theorem uniformity_dist' :
    𝓤 (Completion α) = ⨅ ε : { ε : ℝ // 0 < ε }, 𝓟 { p | dist p.1 p.2 < ε.val } := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ Eq (uniformity (UniformSpace.Completion α)) (iInf fun ε => Filter.principal  …
  -/
  ext s; rw [mem_iInf_of_directed]
    /-
      case h
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ⊢ Iff (Membership.mem (uniformity (UniformSpace.Completion α)) s) (Exists fun  …
    -/
  · simp [Completion.mem_uniformity_dist, subset_def]
    /-
      🎉 no goals
    -/
    /-
      case h.h
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      ⊢ Directed (fun x1 x2 => GE.ge x1 x2) fun ε => Filter.principal (setOf fun p = …
    -/
  · rintro ⟨r, hr⟩ ⟨p, hp⟩
    /-
      case h.h.mk.mk
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      r : Real
      hr : LT.lt 0 r
      p : Real
      hp : LT.lt 0 p
      ⊢ Exists fun z => And ((fun x1 x2 => GE.ge x1 x2) ((fun ε => Filter.principal  …
    -/
    use ⟨min r p, lt_min hr hp⟩
    /-
      case h
      α : Type u
      inst✝ : PseudoMetricSpace α
      s : Set (Prod (UniformSpace.Completion α) (UniformSpace.Completion α))
      r : Real
      hr : LT.lt 0 r
      p : Real
      hp : LT.lt 0 p
      ⊢ And ((fun x1 x2 => GE.ge x1 x2) ((fun ε => Filter.principal (setOf fun p =>  …
    -/
    simp +contextual [lt_min_iff]
    /-
      🎉 no goals
    -/


protected theorem uniformity_dist : 𝓤 (Completion α) = ⨅ ε > 0, 𝓟 { p | dist p.1 p.2 < ε } := by
  /-
    α : Type u
    inst✝ : PseudoMetricSpace α
    ⊢ Eq (uniformity (UniformSpace.Completion α)) (iInf fun ε => iInf fun h => Fil …
  -/
  simpa [iInf_subtype] using @Completion.uniformity_dist' α _
  /-
    🎉 no goals
  -/


/-- Metric space structure on the completion of a pseudo_metric space. -/
instance instMetricSpace : MetricSpace (Completion α) :=
  @MetricSpace.ofT0PseudoMetricSpace _
    { dist_self := Completion.dist_self
      dist_comm := Completion.dist_comm
      dist_triangle := Completion.dist_triangle
      dist := dist
      toUniformSpace := inferInstance
      uniformity_dist := Completion.uniformity_dist } _


@[deprecated eq_of_dist_eq_zero (since := "2024-03-10")]
protected theorem eq_of_dist_eq_zero (x y : Completion α) (h : dist x y = 0) : x = y :=
  eq_of_dist_eq_zero h


/-- The embedding of a metric space in its completion is an isometry. -/
theorem coe_isometry : Isometry ((↑) : α → Completion α) :=
  Isometry.of_dist_eq Completion.dist_eq


@[simp]
protected theorem edist_eq (x y : α) : edist (x : Completion α) y = edist x y :=
  coe_isometry x y


instance {M} [Zero M] [Zero α] [SMul M α] [PseudoMetricSpace M] [BoundedSMul M α] :
    BoundedSMul M (Completion α) where
  dist_smul_pair' c x₁ x₂ := by
    induction x₁, x₂ using induction_on₂ with
    | hp =>
      exact isClosed_le
        ((continuous_fst.const_smul _).dist (continuous_snd.const_smul _))
        (continuous_const.mul (continuous_fst.dist continuous_snd))
    | ih x₁ x₂ =>
      rw [← coe_smul, ← coe_smul, Completion.dist_eq,  Completion.dist_eq]
      exact dist_smul_pair c x₁ x₂
  dist_pair_smul' c₁ c₂ x := by
    induction x using induction_on with
    | hp =>
      exact isClosed_le
        ((continuous_const_smul _).dist (continuous_const_smul _))
        (continuous_const.mul (continuous_id.dist continuous_const))
    | ih x =>
      rw [← coe_smul, ← coe_smul, Completion.dist_eq, ← coe_zero, Completion.dist_eq]
      exact dist_pair_smul c₁ c₂ x


theorem LipschitzWith.completion_extension [MetricSpace β] [CompleteSpace β] {f : α → β}
    {K : ℝ≥0} (h : LipschitzWith K f) : LipschitzWith K (Completion.extension f) :=
  LipschitzWith.of_dist_le_mul fun x y => induction_on₂ x y
                     /-
                       α : Type u
                       β : Type v
                       inst✝² : PseudoMetricSpace α
                       inst✝¹ : MetricSpace β
                       inst✝ : CompleteSpace β
                       f : α → β
                       K : NNReal
                       h : LipschitzWith K f
                       x y : UniformSpace.Completion α
                       ⊢ Continuous fun x => Dist.dist (UniformSpace.Completion.extension f x.1) (Uni …
                     -/
                     /-
                       🎉 no goals
                     -/
    (isClosed_le (by fun_prop) (by fun_prop)) <| by
                                   /-
                                     🎉 no goals
                                   -/
      /-
        α : Type u
        β : Type v
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MetricSpace β
        inst✝ : CompleteSpace β
        f : α → β
        K : NNReal
        h : LipschitzWith K f
        x y : UniformSpace.Completion α
        ⊢ ∀ (a b : α), LE.le (Dist.dist (UniformSpace.Completion.extension f (↑α a)) ( …
      -/
      simpa only [extension_coe h.uniformContinuous, Completion.dist_eq] using h.dist_le_mul
      /-
        🎉 no goals
      -/


theorem LipschitzWith.completion_map [PseudoMetricSpace β] {f : α → β} {K : ℝ≥0}
    (h : LipschitzWith K f) : LipschitzWith K (Completion.map f) :=
  one_mul K ▸ (coe_isometry.lipschitz.comp h).completion_extension


theorem Isometry.completion_extension [MetricSpace β] [CompleteSpace β] {f : α → β}
    (h : Isometry f) : Isometry (Completion.extension f) :=
  Isometry.of_dist_eq fun x y => induction_on₂ x y
                     /-
                       α : Type u
                       β : Type v
                       inst✝² : PseudoMetricSpace α
                       inst✝¹ : MetricSpace β
                       inst✝ : CompleteSpace β
                       f : α → β
                       h : Isometry f
                       x y : UniformSpace.Completion α
                       ⊢ Continuous fun x => Dist.dist (UniformSpace.Completion.extension f x.1) (Uni …
                     -/
                     /-
                       🎉 no goals
                     -/
    (isClosed_eq (by fun_prop) (by fun_prop)) fun _ _ ↦ by
                                   /-
                                     🎉 no goals
                                   -/
      /-
        α : Type u
        β : Type v
        inst✝² : PseudoMetricSpace α
        inst✝¹ : MetricSpace β
        inst✝ : CompleteSpace β
        f : α → β
        h : Isometry f
        x y : UniformSpace.Completion α
        x✝¹ x✝ : α
        ⊢ Eq (Dist.dist (UniformSpace.Completion.extension f (↑α x✝¹)) (UniformSpace.C …
      -/
      simp only [extension_coe h.uniformContinuous, Completion.dist_eq, h.dist_eq]
      /-
        🎉 no goals
      -/


theorem Isometry.completion_map [PseudoMetricSpace β] {f : α → β}
    (h : Isometry f) : Isometry (Completion.map f) :=
  (coe_isometry.comp h).completion_extension

