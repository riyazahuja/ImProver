/-- A finite product of pseudometric spaces is a pseudometric space, with the sup distance. -/
instance pseudoMetricSpacePi : PseudoMetricSpace (∀ b, π b) := by
  /- we construct the instance from the pseudoemetric space instance to avoid checking again that
    the uniformity is the same as the product uniformity, but we register nevertheless a nice
    formula for the distance -/
  let i := PseudoEMetricSpace.toPseudoMetricSpaceOfDist
    (fun f g : ∀ b, π b => ((sup univ fun b => nndist (f b) (g b) : ℝ≥0) : ℝ))
    (fun f g => ((Finset.sup_lt_iff bot_lt_top).2 fun b _ => edist_lt_top _ _).ne)
    (fun f g => by
      simp only [edist_pi_def, edist_nndist, ← ENNReal.coe_finset_sup, ENNReal.coe_toReal])
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    i : PseudoMetricSpace ((b : β) → π b) := PseudoEMetricSpace.toPseudoMetricSpac …
    ⊢ PseudoMetricSpace ((b : β) → π b)
  -/
  refine i.replaceBornology fun s => ?_
  simp only [← isBounded_def, isBounded_iff_eventually, ← forall_isBounded_image_eval_iff,
    forall_mem_image, ← Filter.eventually_all, Function.eval_apply, @dist_nndist (π _)]
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    i : PseudoMetricSpace ((b : β) → π b) := PseudoEMetricSpace.toPseudoMetricSpac …
    s : Set ((b : β) → π b)
    ⊢ Iff (Filter.Eventually (fun x => ∀ (i : β) ⦃x_1 : (x : β) → π x⦄, Membership …
  -/
  refine eventually_congr ((eventually_ge_atTop 0).mono fun C hC ↦ ?_)
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    i : PseudoMetricSpace ((b : β) → π b) := PseudoEMetricSpace.toPseudoMetricSpac …
    s : Set ((b : β) → π b)
    C : Real
    hC : LE.le 0 C
    ⊢ Iff (∀ (i : β) ⦃x : (x : β) → π x⦄, Membership.mem s x → ∀ ⦃x_1 : (x : β) →  …
  -/
  lift C to ℝ≥0 using hC
  refine ⟨fun H x hx y hy ↦ NNReal.coe_le_coe.2 <| Finset.sup_le fun b _ ↦ H b hx hy,
    fun H b x hx y hy ↦ NNReal.coe_le_coe.2 ?_⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoMetricSpace α
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    i : PseudoMetricSpace ((b : β) → π b) := PseudoEMetricSpace.toPseudoMetricSpac …
    s : Set ((b : β) → π b)
    C : NNReal
    H : ∀ ⦃x : (b : β) → π b⦄, Membership.mem s x → ∀ ⦃y : (b : β) → π b⦄, Members …
    b : β
    x : (x : β) → π x
    hx : Membership.mem s x
    y : (x : β) → π x
    hy : Membership.mem s y
    ⊢ LE.le (NNDist.nndist (Function.eval b x) (Function.eval b y)) C
  -/
  simpa only using Finset.sup_le_iff.1 (NNReal.coe_le_coe.1 <| H hx hy) b (Finset.mem_univ b)
  /-
    🎉 no goals
  -/


lemma nndist_pi_def (f g : ∀ b, π b) : nndist f g = sup univ fun b => nndist (f b) (g b) := rfl


lemma dist_pi_def (f g : ∀ b, π b) : dist f g = (sup univ fun b => nndist (f b) (g b) : ℝ≥0) := rfl


lemma nndist_pi_le_iff {f g : ∀ b, π b} {r : ℝ≥0} :
                                                       /-
                                                         β : Type u_2
                                                         π : β → Type u_3
                                                         inst✝¹ : Fintype β
                                                         inst✝ : (b : β) → PseudoMetricSpace (π b)
                                                         f g : (b : β) → π b
                                                         r : NNReal
                                                         ⊢ Iff (LE.le (NNDist.nndist f g) r) (∀ (b : β), LE.le (NNDist.nndist (f b) (g  …
                                                       -/
    nndist f g ≤ r ↔ ∀ b, nndist (f b) (g b) ≤ r := by simp [nndist_pi_def]
                                                       /-
                                                         🎉 no goals
                                                       -/


lemma nndist_pi_lt_iff {f g : ∀ b, π b} {r : ℝ≥0} (hr : 0 < r) :
    nndist f g < r ↔ ∀ b, nndist (f b) (g b) < r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    ⊢ Iff (LT.lt (NNDist.nndist f g) r) (∀ (b : β), LT.lt (NNDist.nndist (f b) (g  …
  -/
  rw [← bot_eq_zero'] at hr
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt Bot.bot r
    ⊢ Iff (LT.lt (NNDist.nndist f g) r) (∀ (b : β), LT.lt (NNDist.nndist (f b) (g  …
  -/
  simp [nndist_pi_def, Finset.sup_lt_iff hr]
  /-
    🎉 no goals
  -/


lemma nndist_pi_eq_iff {f g : ∀ b, π b} {r : ℝ≥0} (hr : 0 < r) :
    nndist f g = r ↔ (∃ i, nndist (f i) (g i) = r) ∧ ∀ b, nndist (f b) (g b) ≤ r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    ⊢ Iff (Eq (NNDist.nndist f g) r) (And (Exists fun i => Eq (NNDist.nndist (f i) …
  -/
  rw [eq_iff_le_not_lt, nndist_pi_lt_iff hr, nndist_pi_le_iff, not_forall, and_comm]
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    ⊢ Iff (And (Exists fun x => Not (LT.lt (NNDist.nndist (f x) (g x)) r)) (∀ (b : …
  -/
  simp_rw [not_lt, and_congr_left_iff, le_antisymm_iff]
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    ⊢ (∀ (b : β), LE.le (NNDist.nndist (f b) (g b)) r) → Iff (Exists fun x => LE.l …
  -/
  intro h
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (b : β), LE.le (NNDist.nndist (f b) (g b)) r
    ⊢ Iff (Exists fun x => LE.le r (NNDist.nndist (f x) (g x))) (Exists fun i => A …
  -/
  refine exists_congr fun b => ?_
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 r
    h : ∀ (b : β), LE.le (NNDist.nndist (f b) (g b)) r
    b : β
    ⊢ Iff (LE.le r (NNDist.nndist (f b) (g b))) (And (LE.le (NNDist.nndist (f b) ( …
  -/
  apply (and_iff_right <| h _).symm
  /-
    🎉 no goals
  -/


lemma dist_pi_lt_iff {f g : ∀ b, π b} {r : ℝ} (hr : 0 < r) :
    dist f g < r ↔ ∀ b, dist (f b) (g b) < r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : Real
    hr : LT.lt 0 r
    ⊢ Iff (LT.lt (Dist.dist f g) r) (∀ (b : β), LT.lt (Dist.dist (f b) (g b)) r)
  -/
  lift r to ℝ≥0 using hr.le
  /-
    case intro
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 ↑r
    ⊢ Iff (LT.lt (Dist.dist f g) ↑r) (∀ (b : β), LT.lt (Dist.dist (f b) (g b)) ↑r)
  -/
  exact nndist_pi_lt_iff hr
  /-
    🎉 no goals
  -/


lemma dist_pi_le_iff {f g : ∀ b, π b} {r : ℝ} (hr : 0 ≤ r) :
    dist f g ≤ r ↔ ∀ b, dist (f b) (g b) ≤ r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : Real
    hr : LE.le 0 r
    ⊢ Iff (LE.le (Dist.dist f g) r) (∀ (b : β), LE.le (Dist.dist (f b) (g b)) r)
  -/
  lift r to ℝ≥0 using hr
  /-
    case intro
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    ⊢ Iff (LE.le (Dist.dist f g) ↑r) (∀ (b : β), LE.le (Dist.dist (f b) (g b)) ↑r)
  -/
  exact nndist_pi_le_iff
  /-
    🎉 no goals
  -/


lemma dist_pi_eq_iff {f g : ∀ b, π b} {r : ℝ} (hr : 0 < r) :
    dist f g = r ↔ (∃ i, dist (f i) (g i) = r) ∧ ∀ b, dist (f b) (g b) ≤ r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : Real
    hr : LT.lt 0 r
    ⊢ Iff (Eq (Dist.dist f g) r) (And (Exists fun i => Eq (Dist.dist (f i) (g i))  …
  -/
  lift r to ℝ≥0 using hr.le
  /-
    case intro
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    r : NNReal
    hr : LT.lt 0 ↑r
    ⊢ Iff (Eq (Dist.dist f g) ↑r) (And (Exists fun i => Eq (Dist.dist (f i) (g i)) …
  -/
  simp_rw [← coe_nndist, NNReal.coe_inj, nndist_pi_eq_iff hr, NNReal.coe_le_coe]
  /-
    🎉 no goals
  -/


lemma dist_pi_le_iff' [Nonempty β] {f g : ∀ b, π b} {r : ℝ} :
    dist f g ≤ r ↔ ∀ b, dist (f b) (g b) ≤ r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝² : Fintype β
    inst✝¹ : (b : β) → PseudoMetricSpace (π b)
    inst✝ : Nonempty β
    f g : (b : β) → π b
    r : Real
    ⊢ Iff (LE.le (Dist.dist f g) r) (∀ (b : β), LE.le (Dist.dist (f b) (g b)) r)
  -/
  by_cases hr : 0 ≤ r
    /-
      case pos
      β : Type u_2
      π : β → Type u_3
      inst✝² : Fintype β
      inst✝¹ : (b : β) → PseudoMetricSpace (π b)
      inst✝ : Nonempty β
      f g : (b : β) → π b
      r : Real
      hr : LE.le 0 r
      ⊢ Iff (LE.le (Dist.dist f g) r) (∀ (b : β), LE.le (Dist.dist (f b) (g b)) r)
    -/
  · exact dist_pi_le_iff hr
    /-
      🎉 no goals
    -/
  · exact iff_of_false (fun h => hr <| dist_nonneg.trans h) fun h =>
      hr <| dist_nonneg.trans <| h <| Classical.arbitrary _


lemma dist_pi_const_le (a b : α) : (dist (fun _ : β => a) fun _ => b) ≤ dist a b :=
  (dist_pi_le_iff dist_nonneg).2 fun _ => le_rfl


lemma nndist_pi_const_le (a b : α) : (nndist (fun _ : β => a) fun _ => b) ≤ nndist a b :=
  nndist_pi_le_iff.2 fun _ => le_rfl


@[simp]
lemma dist_pi_const [Nonempty β] (a b : α) : (dist (fun _ : β => a) fun _ => b) = dist a b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : PseudoMetricSpace α
    inst✝¹ : Fintype β
    inst✝ : Nonempty β
    a b : α
    ⊢ Eq (Dist.dist (fun x => a) fun x => b) (Dist.dist a b)
  -/
  simpa only [dist_edist] using congr_arg ENNReal.toReal (edist_pi_const a b)
  /-
    🎉 no goals
  -/


@[simp]
lemma nndist_pi_const [Nonempty β] (a b : α) : (nndist (fun _ : β => a) fun _ => b) = nndist a b :=
  NNReal.eq <| dist_pi_const a b


lemma nndist_le_pi_nndist (f g : ∀ b, π b) (b : β) : nndist (f b) (g b) ≤ nndist f g := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    b : β
    ⊢ LE.le (NNDist.nndist (f b) (g b)) (NNDist.nndist f g)
  -/
  rw [← ENNReal.coe_le_coe, ← edist_nndist, ← edist_nndist]
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    b : β
    ⊢ LE.le (EDist.edist (f b) (g b)) (EDist.edist f g)
  -/
  exact edist_le_pi_edist f g b
  /-
    🎉 no goals
  -/


lemma dist_le_pi_dist (f g : ∀ b, π b) (b : β) : dist (f b) (g b) ≤ dist f g := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    f g : (b : β) → π b
    b : β
    ⊢ LE.le (Dist.dist (f b) (g b)) (Dist.dist f g)
  -/
  simp only [dist_nndist, NNReal.coe_le_coe, nndist_le_pi_nndist f g b]
  /-
    🎉 no goals
  -/


/-- An open ball in a product space is a product of open balls. See also `ball_pi'`
for a version assuming `Nonempty β` instead of `0 < r`. -/
lemma ball_pi (x : ∀ b, π b) {r : ℝ} (hr : 0 < r) :
    ball x r = Set.pi univ fun b => ball (x b) r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LT.lt 0 r
    ⊢ Eq (Metric.ball x r) (Set.univ.pi fun b => Metric.ball (x b) r)
  -/
  ext p
  /-
    case h
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LT.lt 0 r
    p : (b : β) → π b
    ⊢ Iff (Membership.mem (Metric.ball x r) p) (Membership.mem (Set.univ.pi fun b  …
  -/
  simp [dist_pi_lt_iff hr]
  /-
    🎉 no goals
  -/


/-- An open ball in a product space is a product of open balls. See also `ball_pi`
for a version assuming `0 < r` instead of `Nonempty β`. -/
lemma ball_pi' [Nonempty β] (x : ∀ b, π b) (r : ℝ) :
    ball x r = Set.pi univ fun b => ball (x b) r :=
                                               /-
                                                 β : Type u_2
                                                 π : β → Type u_3
                                                 inst✝² : Fintype β
                                                 inst✝¹ : (b : β) → PseudoMetricSpace (π b)
                                                 inst✝ : Nonempty β
                                                 x : (b : β) → π b
                                                 r : Real
                                                 hr : LE.le r 0
                                                 ⊢ Eq (Metric.ball x r) (Set.univ.pi fun b => Metric.ball (x b) r)
                                               -/
  (lt_or_le 0 r).elim (ball_pi x) fun hr => by simp [ball_eq_empty.2 hr]
                                               /-
                                                 🎉 no goals
                                               -/


/-- A closed ball in a product space is a product of closed balls. See also `closedBall_pi'`
for a version assuming `Nonempty β` instead of `0 ≤ r`. -/
lemma closedBall_pi (x : ∀ b, π b) {r : ℝ} (hr : 0 ≤ r) :
    closedBall x r = Set.pi univ fun b => closedBall (x b) r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LE.le 0 r
    ⊢ Eq (Metric.closedBall x r) (Set.univ.pi fun b => Metric.closedBall (x b) r)
  -/
  ext p
  /-
    case h
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    x : (b : β) → π b
    r : Real
    hr : LE.le 0 r
    p : (b : β) → π b
    ⊢ Iff (Membership.mem (Metric.closedBall x r) p) (Membership.mem (Set.univ.pi  …
  -/
  simp [dist_pi_le_iff hr]
  /-
    🎉 no goals
  -/


/-- A closed ball in a product space is a product of closed balls. See also `closedBall_pi`
for a version assuming `0 ≤ r` instead of `Nonempty β`. -/
lemma closedBall_pi' [Nonempty β] (x : ∀ b, π b) (r : ℝ) :
    closedBall x r = Set.pi univ fun b => closedBall (x b) r :=
                                                     /-
                                                       β : Type u_2
                                                       π : β → Type u_3
                                                       inst✝² : Fintype β
                                                       inst✝¹ : (b : β) → PseudoMetricSpace (π b)
                                                       inst✝ : Nonempty β
                                                       x : (b : β) → π b
                                                       r : Real
                                                       hr : LT.lt r 0
                                                       ⊢ Eq (Metric.closedBall x r) (Set.univ.pi fun b => Metric.closedBall (x b) r)
                                                     -/
  (le_or_lt 0 r).elim (closedBall_pi x) fun hr => by simp [closedBall_eq_empty.2 hr]
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- A sphere in a product space is a union of spheres on each component restricted to the closed
ball. -/
lemma sphere_pi (x : ∀ b, π b) {r : ℝ} (h : 0 < r ∨ Nonempty β) :
    sphere x r = (⋃ i : β, Function.eval i ⁻¹' sphere (x i) r) ∩ closedBall x r := by
  /-
    β : Type u_2
    π : β → Type u_3
    inst✝¹ : Fintype β
    inst✝ : (b : β) → PseudoMetricSpace (π b)
    x : (b : β) → π b
    r : Real
    h : Or (LT.lt 0 r) (Nonempty β)
    ⊢ Eq (Metric.sphere x r) (Inter.inter (Set.iUnion fun i => Set.preimage (Funct …
  -/
  obtain hr | rfl | hr := lt_trichotomy r 0
    /-
      case inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      r : Real
      h : Or (LT.lt 0 r) (Nonempty β)
      hr : LT.lt r 0
      ⊢ Eq (Metric.sphere x r) (Inter.inter (Set.iUnion fun i => Set.preimage (Funct …
    -/
  · simp [hr]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      ⊢ Eq (Metric.sphere x 0) (Inter.inter (Set.iUnion fun i => Set.preimage (Funct …
    -/
  · rw [closedBall_eq_sphere_of_nonpos le_rfl, eq_comm, Set.inter_eq_right]
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      ⊢ HasSubset.Subset (Metric.sphere x 0) (Set.iUnion fun i => Set.preimage (Func …
    -/
    letI := h.resolve_left (lt_irrefl _)
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      ⊢ HasSubset.Subset (Metric.sphere x 0) (Set.iUnion fun i => Set.preimage (Func …
    -/
    inhabit β
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      inhabited_h : Inhabited β
      ⊢ HasSubset.Subset (Metric.sphere x 0) (Set.iUnion fun i => Set.preimage (Func …
    -/
    refine subset_iUnion_of_subset default ?_
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      inhabited_h : Inhabited β
      ⊢ HasSubset.Subset (Metric.sphere x 0) (Set.preimage (Function.eval Inhabited. …
    -/
    intro x hx
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x✝ : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      inhabited_h : Inhabited β
      x : (x : β) → π x
      hx : Membership.mem (Metric.sphere x✝ 0) x
      ⊢ Membership.mem (Set.preimage (Function.eval Inhabited.default) (Metric.spher …
    -/
    replace hx := hx.le
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x✝ : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      inhabited_h : Inhabited β
      x : (x : β) → π x
      hx : LE.le (Dist.dist x x✝) 0
      ⊢ Membership.mem (Set.preimage (Function.eval Inhabited.default) (Metric.spher …
    -/
    rw [dist_pi_le_iff le_rfl] at hx
    /-
      case inr.inl
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x✝ : (b : β) → π b
      h : Or (LT.lt 0 0) (Nonempty β)
      this : Nonempty β := Or.resolve_left h (lt_irrefl 0)
      inhabited_h : Inhabited β
      x : (x : β) → π x
      hx : ∀ (b : β), LE.le (Dist.dist (x b) (x✝ b)) 0
      ⊢ Membership.mem (Set.preimage (Function.eval Inhabited.default) (Metric.spher …
    -/
    exact le_antisymm (hx default) dist_nonneg
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      r : Real
      h : Or (LT.lt 0 r) (Nonempty β)
      hr : LT.lt 0 r
      ⊢ Eq (Metric.sphere x r) (Inter.inter (Set.iUnion fun i => Set.preimage (Funct …
    -/
  · ext
    /-
      case inr.inr.h
      β : Type u_2
      π : β → Type u_3
      inst✝¹ : Fintype β
      inst✝ : (b : β) → PseudoMetricSpace (π b)
      x : (b : β) → π b
      r : Real
      h : Or (LT.lt 0 r) (Nonempty β)
      hr : LT.lt 0 r
      x✝ : (b : β) → π b
      ⊢ Iff (Membership.mem (Metric.sphere x r) x✝) (Membership.mem (Inter.inter (Se …
    -/
    simp [dist_pi_eq_iff hr, dist_pi_le_iff hr.le]
    /-
      🎉 no goals
    -/


@[simp]
lemma Fin.nndist_insertNth_insertNth {n : ℕ} {α : Fin (n + 1) → Type*}
    [∀ i, PseudoMetricSpace (α i)] (i : Fin (n + 1)) (x y : α i) (f g : ∀ j, α (i.succAbove j)) :
    nndist (i.insertNth x f) (i.insertNth y g) = max (nndist x y) (nndist f g) :=
                                  /-
                                    n : Nat
                                    α : Fin (HAdd.hAdd n 1) → Type u_4
                                    inst✝ : (i : Fin (HAdd.hAdd n 1)) → PseudoMetricSpace (α i)
                                    i : Fin (HAdd.hAdd n 1)
                                    x y : α i
                                    f g : (j : Fin n) → α (i.succAbove j)
                                    c : NNReal
                                    ⊢ Iff (LE.le (NNDist.nndist (i.insertNth x f) (i.insertNth y g)) c) (LE.le (Ma …
                                  -/
  eq_of_forall_ge_iff fun c => by simp [nndist_pi_le_iff, i.forall_iff_succAbove]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
lemma Fin.dist_insertNth_insertNth {n : ℕ} {α : Fin (n + 1) → Type*}
    [∀ i, PseudoMetricSpace (α i)] (i : Fin (n + 1)) (x y : α i) (f g : ∀ j, α (i.succAbove j)) :
    dist (i.insertNth x f) (i.insertNth y g) = max (dist x y) (dist f g) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_4
    inst✝ : (i : Fin (HAdd.hAdd n 1)) → PseudoMetricSpace (α i)
    i : Fin (HAdd.hAdd n 1)
    x y : α i
    f g : (j : Fin n) → α (i.succAbove j)
    ⊢ Eq (Dist.dist (i.insertNth x f) (i.insertNth y g)) (Max.max (Dist.dist x y)  …
  -/
  simp only [dist_nndist, Fin.nndist_insertNth_insertNth, NNReal.coe_max]
  /-
    🎉 no goals
  -/

