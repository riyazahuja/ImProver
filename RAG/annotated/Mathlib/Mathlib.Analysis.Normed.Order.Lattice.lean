/-- Let `α` be an `AddCommGroup` with a `Lattice` structure. A norm on `α` is *solid* if, for `a`
and `b` in `α`, with absolute values `|a|` and `|b|` respectively, `|a| ≤ |b|` implies `‖a‖ ≤ ‖b‖`.
-/
class HasSolidNorm (α : Type*) [NormedAddCommGroup α] [Lattice α] : Prop where
  solid : ∀ ⦃x y : α⦄, |x| ≤ |y| → ‖x‖ ≤ ‖y‖


theorem norm_le_norm_of_abs_le_abs {a b : α} (h : |a| ≤ |b|) : ‖a‖ ≤ ‖b‖ :=
  HasSolidNorm.solid h


/-- If `α` has a solid norm, then the balls centered at the origin of `α` are solid sets. -/
theorem LatticeOrderedAddCommGroup.isSolid_ball (r : ℝ) :
    LatticeOrderedAddCommGroup.IsSolid (Metric.ball (0 : α) r) := fun _ hx _ hxy =>
  mem_ball_zero_iff.mpr ((HasSolidNorm.solid hxy).trans_lt (mem_ball_zero_iff.mp hx))


instance : HasSolidNorm ℝ := ⟨fun _ _ => id⟩


                                              /-
                                                α : Type u_1
                                                inst✝² : NormedAddCommGroup α
                                                inst✝¹ : Lattice α
                                                inst✝ : HasSolidNorm α
                                                x✝² x✝¹ : Rat
                                                x✝ : LE.le (abs x✝²) (abs x✝¹)
                                                ⊢ LE.le (Norm.norm x✝²) (Norm.norm x✝¹)
                                              -/
instance : HasSolidNorm ℚ := ⟨fun _ _ _ => by simpa only [norm, ← Rat.cast_abs, Rat.cast_le]⟩
                                              /-
                                                🎉 no goals
                                              -/


/--
Let `α` be a normed commutative group equipped with a partial order covariant with addition, with
respect which `α` forms a lattice. Suppose that `α` is *solid*, that is to say, for `a` and `b` in
`α`, with absolute values `|a|` and `|b|` respectively, `|a| ≤ |b|` implies `‖a‖ ≤ ‖b‖`. Then `α` is
said to be a normed lattice ordered group.
-/
class NormedLatticeAddCommGroup (α : Type*) extends
    NormedAddCommGroup α, Lattice α, HasSolidNorm α where
  add_le_add_left : ∀ a b : α, a ≤ b → ∀ c : α, c + a ≤ c + b


instance Int.normedLatticeAddCommGroup : NormedLatticeAddCommGroup ℤ where
                    /-
                      x y : Int
                      h : LE.le (abs x) (abs y)
                      ⊢ LE.le (Norm.norm x) (Norm.norm y)
                    -/
  solid x y h := by simpa [← Int.norm_cast_real, ← Int.cast_abs] using h
                    /-
                      🎉 no goals
                    -/
  add_le_add_left _ _ := add_le_add_left


instance Rat.normedLatticeAddCommGroup : NormedLatticeAddCommGroup ℚ where
                    /-
                      x y : Rat
                      h : LE.le (abs x) (abs y)
                      ⊢ LE.le (Norm.norm x) (Norm.norm y)
                    -/
  solid x y h := by simpa [← Rat.norm_cast_real, ← Rat.cast_abs] using h
                    /-
                      🎉 no goals
                    -/
  add_le_add_left _ _ := add_le_add_left


instance Real.normedLatticeAddCommGroup : NormedLatticeAddCommGroup ℝ where
  add_le_add_left _ _ h _ := add_le_add le_rfl h

-- see Note [lower instance priority]

/-- A normed lattice ordered group is an ordered additive commutative group
-/
instance (priority := 100) NormedLatticeAddCommGroup.toOrderedAddCommGroup {α : Type*}
    [h : NormedLatticeAddCommGroup α] : OrderedAddCommGroup α :=
  { h with }


theorem dual_solid (a b : α) (h : b ⊓ -b ≤ a ⊓ -a) : ‖a‖ ≤ ‖b‖ := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Norm.norm a) (Norm.norm b)
  -/
  apply solid
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (abs a) (abs b)
  -/
  rw [abs]
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Max.max a (Neg.neg a)) (abs b)
  -/
  nth_rw 1 [← neg_neg a]
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Max.max (Neg.neg (Neg.neg a)) (Neg.neg a)) (abs b)
  -/
  rw [← neg_inf]
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Neg.neg (Min.min (Neg.neg a) a)) (abs b)
  -/
  rw [abs]
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Neg.neg (Min.min (Neg.neg a) a)) (Max.max b (Neg.neg b))
  -/
  nth_rw 1 [← neg_neg b]
  /-
    case a
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b : α
    h : LE.le (Min.min b (Neg.neg b)) (Min.min a (Neg.neg a))
    ⊢ LE.le (Neg.neg (Min.min (Neg.neg a) a)) (Max.max (Neg.neg (Neg.neg b)) (Neg. …
  -/
  rwa [← neg_inf, neg_le_neg_iff, inf_comm _ b, inf_comm _ a]
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- Let `α` be a normed lattice ordered group, then the order dual is also a
normed lattice ordered group.
-/
instance (priority := 100) OrderDual.instNormedLatticeAddCommGroup :
    NormedLatticeAddCommGroup αᵒᵈ :=
  { OrderDual.orderedAddCommGroup, OrderDual.normedAddCommGroup, OrderDual.instLattice α with
    solid := dual_solid (α := α) }


theorem norm_abs_eq_norm (a : α) : ‖|a|‖ = ‖a‖ :=
  (solid (abs_abs a).le).antisymm (solid (abs_abs a).symm.le)


theorem norm_inf_sub_inf_le_add_norm (a b c d : α) : ‖a ⊓ b - c ⊓ d‖ ≤ ‖a - c‖ + ‖b - d‖ := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (Norm.norm (HSub.hSub (Min.min a b) (Min.min c d))) (HAdd.hAdd (Norm.n …
  -/
  rw [← norm_abs_eq_norm (a - c), ← norm_abs_eq_norm (b - d)]
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (Norm.norm (HSub.hSub (Min.min a b) (Min.min c d))) (HAdd.hAdd (Norm.n …
  -/
  refine le_trans (solid ?_) (norm_add_le |a - c| |b - d|)
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (abs (HSub.hSub (Min.min a b) (Min.min c d))) (abs (HAdd.hAdd (abs (HS …
  -/
  rw [abs_of_nonneg (add_nonneg (abs_nonneg (a - c)) (abs_nonneg (b - d)))]
  calc
    |a ⊓ b - c ⊓ d| = |a ⊓ b - c ⊓ b + (c ⊓ b - c ⊓ d)| := by rw [sub_add_sub_cancel]
    _ ≤ |a ⊓ b - c ⊓ b| + |c ⊓ b - c ⊓ d| := abs_add_le _ _
    _ ≤ |a - c| + |b - d| := by
      apply add_le_add
      · exact abs_inf_sub_inf_le_abs _ _ _
      · rw [inf_comm c, inf_comm c]
        exact abs_inf_sub_inf_le_abs _ _ _


theorem norm_sup_sub_sup_le_add_norm (a b c d : α) : ‖a ⊔ b - c ⊔ d‖ ≤ ‖a - c‖ + ‖b - d‖ := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (Norm.norm (HSub.hSub (Max.max a b) (Max.max c d))) (HAdd.hAdd (Norm.n …
  -/
  rw [← norm_abs_eq_norm (a - c), ← norm_abs_eq_norm (b - d)]
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (Norm.norm (HSub.hSub (Max.max a b) (Max.max c d))) (HAdd.hAdd (Norm.n …
  -/
  refine le_trans (solid ?_) (norm_add_le |a - c| |b - d|)
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    a b c d : α
    ⊢ LE.le (abs (HSub.hSub (Max.max a b) (Max.max c d))) (abs (HAdd.hAdd (abs (HS …
  -/
  rw [abs_of_nonneg (add_nonneg (abs_nonneg (a - c)) (abs_nonneg (b - d)))]
  calc
    |a ⊔ b - c ⊔ d| = |a ⊔ b - c ⊔ b + (c ⊔ b - c ⊔ d)| := by rw [sub_add_sub_cancel]
    _ ≤ |a ⊔ b - c ⊔ b| + |c ⊔ b - c ⊔ d| := abs_add_le _ _
    _ ≤ |a - c| + |b - d| := by
      apply add_le_add
      · exact abs_sup_sub_sup_le_abs _ _ _
      · rw [sup_comm c, sup_comm c]
        exact abs_sup_sub_sup_le_abs _ _ _


theorem norm_inf_le_add (x y : α) : ‖x ⊓ y‖ ≤ ‖x‖ + ‖y‖ := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    x y : α
    ⊢ LE.le (Norm.norm (Min.min x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  have h : ‖x ⊓ y - 0 ⊓ 0‖ ≤ ‖x - 0‖ + ‖y - 0‖ := norm_inf_sub_inf_le_add_norm x y 0 0
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    x y : α
    h : LE.le (Norm.norm (HSub.hSub (Min.min x y) (Min.min 0 0))) (HAdd.hAdd (Norm …
    ⊢ LE.le (Norm.norm (Min.min x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  simpa only [inf_idem, sub_zero] using h
  /-
    🎉 no goals
  -/


theorem norm_sup_le_add (x y : α) : ‖x ⊔ y‖ ≤ ‖x‖ + ‖y‖ := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    x y : α
    ⊢ LE.le (Norm.norm (Max.max x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  have h : ‖x ⊔ y - 0 ⊔ 0‖ ≤ ‖x - 0‖ + ‖y - 0‖ := norm_sup_sub_sup_le_add_norm x y 0 0
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    x y : α
    h : LE.le (Norm.norm (HSub.hSub (Max.max x y) (Max.max 0 0))) (HAdd.hAdd (Norm …
    ⊢ LE.le (Norm.norm (Max.max x y)) (HAdd.hAdd (Norm.norm x) (Norm.norm y))
  -/
  simpa only [sup_idem, sub_zero] using h
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

/-- Let `α` be a normed lattice ordered group. Then the infimum is jointly continuous.
-/
instance (priority := 100) NormedLatticeAddCommGroup.continuousInf : ContinuousInf α := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    ⊢ ContinuousInf α
  -/
  refine ⟨continuous_iff_continuousAt.2 fun q => tendsto_iff_norm_sub_tendsto_zero.2 <| ?_⟩
  have : ∀ p : α × α, ‖p.1 ⊓ p.2 - q.1 ⊓ q.2‖ ≤ ‖p.1 - q.1‖ + ‖p.2 - q.2‖ := fun _ =>
    norm_inf_sub_inf_le_add_norm _ _ _ _
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    q : Prod α α
    this : ∀ (p : Prod α α), LE.le (Norm.norm (HSub.hSub (Min.min p.1 p.2) (Min.mi …
    ⊢ Filter.Tendsto (fun e => Norm.norm (HSub.hSub (Min.min e.1 e.2) ((fun p => M …
  -/
  refine squeeze_zero (fun e => norm_nonneg _) this ?_
  convert ((continuous_fst.tendsto q).sub <| tendsto_const_nhds).norm.add
    ((continuous_snd.tendsto q).sub <| tendsto_const_nhds).norm
  /-
    case h.e'_5.h.e'_3
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    q : Prod α α
    this : ∀ (p : Prod α α), LE.le (Norm.norm (HSub.hSub (Min.min p.1 p.2) (Min.mi …
    ⊢ Eq 0 (HAdd.hAdd (Norm.norm (HSub.hSub q.1 q.1)) (Norm.norm (HSub.hSub q.2 q. …
  -/
  simp
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) NormedLatticeAddCommGroup.continuousSup {α : Type*}
    [NormedLatticeAddCommGroup α] : ContinuousSup α :=
  OrderDual.continuousSup αᵒᵈ

-- see Note [lower instance priority]

/--
Let `α` be a normed lattice ordered group. Then `α` is a topological lattice in the norm topology.
-/
instance (priority := 100) NormedLatticeAddCommGroup.toTopologicalLattice : TopologicalLattice α :=
  TopologicalLattice.mk


theorem norm_abs_sub_abs (a b : α) : ‖|a| - |b|‖ ≤ ‖a - b‖ := solid (abs_abs_sub_abs_le _ _)


theorem norm_sup_sub_sup_le_norm (x y z : α) : ‖x ⊔ z - y ⊔ z‖ ≤ ‖x - y‖ :=
  solid (abs_sup_sub_sup_le_abs x y z)


theorem norm_inf_sub_inf_le_norm (x y z : α) : ‖x ⊓ z - y ⊓ z‖ ≤ ‖x - y‖ :=
  solid (abs_inf_sub_inf_le_abs x y z)


theorem lipschitzWith_sup_right (z : α) : LipschitzWith 1 fun x => x ⊔ z :=
  LipschitzWith.of_dist_le_mul fun x y => by
    /-
      α : Type u_1
      inst✝ : NormedLatticeAddCommGroup α
      z x y : α
      ⊢ LE.le (Dist.dist (Max.max x z) (Max.max y z)) (HMul.hMul (↑1) (Dist.dist x y))
    -/
    rw [NNReal.coe_one, one_mul, dist_eq_norm, dist_eq_norm]
    /-
      α : Type u_1
      inst✝ : NormedLatticeAddCommGroup α
      z x y : α
      ⊢ LE.le (Norm.norm (HSub.hSub (Max.max x z) (Max.max y z))) (Norm.norm (HSub.h …
    -/
    exact norm_sup_sub_sup_le_norm x y z
    /-
      🎉 no goals
    -/


lemma lipschitzWith_posPart : LipschitzWith 1 (posPart : α → α) :=
  lipschitzWith_sup_right 0


lemma lipschitzWith_negPart : LipschitzWith 1 (negPart : α → α) := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    ⊢ LipschitzWith 1 NegPart.negPart
  -/
  simpa [Function.comp] using lipschitzWith_posPart.comp LipschitzWith.id.neg
  /-
    🎉 no goals
  -/


@[fun_prop]
lemma continuous_posPart : Continuous (posPart : α → α) := lipschitzWith_posPart.continuous


@[fun_prop]
lemma continuous_negPart : Continuous (negPart : α → α) := lipschitzWith_negPart.continuous


lemma isClosed_nonneg : IsClosed {x : α | 0 ≤ x} := by
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    ⊢ IsClosed (setOf fun x => LE.le 0 x)
  -/
  have : {x : α | 0 ≤ x} = negPart ⁻¹' {0} := by ext; simp [negPart_eq_zero]
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    this : Eq (setOf fun x => LE.le 0 x) (Set.preimage NegPart.negPart (Singleton. …
    ⊢ IsClosed (setOf fun x => LE.le 0 x)
  -/
  rw [this]
  /-
    α : Type u_1
    inst✝ : NormedLatticeAddCommGroup α
    this : Eq (setOf fun x => LE.le 0 x) (Set.preimage NegPart.negPart (Singleton. …
    ⊢ IsClosed (Set.preimage NegPart.negPart (Singleton.singleton 0))
  -/
  exact isClosed_singleton.preimage continuous_negPart
  /-
    🎉 no goals
  -/


theorem isClosed_le_of_isClosed_nonneg {G} [OrderedAddCommGroup G] [TopologicalSpace G]
    [ContinuousSub G] (h : IsClosed { x : G | 0 ≤ x }) :
    IsClosed { p : G × G | p.fst ≤ p.snd } := by
  have : { p : G × G | p.fst ≤ p.snd } = (fun p : G × G => p.snd - p.fst) ⁻¹' { x : G | 0 ≤ x } :=
    by ext1 p; simp only [sub_nonneg, Set.preimage_setOf_eq]
  /-
    G : Type u_2
    inst✝² : OrderedAddCommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousSub G
    h : IsClosed (setOf fun x => LE.le 0 x)
    this : Eq (setOf fun p => LE.le p.1 p.2) (Set.preimage (fun p => HSub.hSub p.2 …
    ⊢ IsClosed (setOf fun p => LE.le p.1 p.2)
  -/
  rw [this]
  /-
    G : Type u_2
    inst✝² : OrderedAddCommGroup G
    inst✝¹ : TopologicalSpace G
    inst✝ : ContinuousSub G
    h : IsClosed (setOf fun x => LE.le 0 x)
    this : Eq (setOf fun p => LE.le p.1 p.2) (Set.preimage (fun p => HSub.hSub p.2 …
    ⊢ IsClosed (Set.preimage (fun p => HSub.hSub p.2 p.1) (setOf fun x => LE.le 0  …
  -/
  exact IsClosed.preimage (continuous_snd.sub continuous_fst) h
  /-
    🎉 no goals
  -/

-- See note [lower instance priority]

instance (priority := 100) NormedLatticeAddCommGroup.orderClosedTopology {E}
    [NormedLatticeAddCommGroup E] : OrderClosedTopology E :=
  ⟨isClosed_le_of_isClosed_nonneg isClosed_nonneg⟩

