/-- Class `LipschitzAdd M` says that the addition `(+) : X × X → X` is Lipschitz jointly in
the two arguments. -/
class LipschitzAdd [AddMonoid β] : Prop where
  lipschitz_add : ∃ C, LipschitzWith C fun p : β × β => p.1 + p.2


/-- Class `LipschitzMul M` says that the multiplication `(*) : X × X → X` is Lipschitz jointly
in the two arguments. -/
@[to_additive]
class LipschitzMul [Monoid β] : Prop where
  lipschitz_mul : ∃ C, LipschitzWith C fun p : β × β => p.1 * p.2


/-- The Lipschitz constant of an `AddMonoid` `β` satisfying `LipschitzAdd` -/
def LipschitzAdd.C [AddMonoid β] [_i : LipschitzAdd β] : ℝ≥0 := Classical.choose _i.lipschitz_add


/-- The Lipschitz constant of a monoid `β` satisfying `LipschitzMul` -/
@[to_additive existing] -- Porting note: had to add `LipschitzAdd.C`. to_additive silently failed
def LipschitzMul.C [_i : LipschitzMul β] : ℝ≥0 := Classical.choose _i.lipschitz_mul


@[to_additive]
theorem lipschitzWith_lipschitz_const_mul_edist [_i : LipschitzMul β] :
    LipschitzWith (LipschitzMul.C β) fun p : β × β => p.1 * p.2 :=
  Classical.choose_spec _i.lipschitz_mul


@[to_additive]
theorem lipschitz_with_lipschitz_const_mul :
    ∀ p q : β × β, dist (p.1 * p.2) (q.1 * q.2) ≤ LipschitzMul.C β * dist p q := by
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    inst✝¹ : Monoid β
    inst✝ : LipschitzMul β
    ⊢ ∀ (p q : Prod β β), LE.le (Dist.dist (HMul.hMul p.1 p.2) (HMul.hMul q.1 q.2) …
  -/
  rw [← lipschitzWith_iff_dist_le_mul]
  /-
    β : Type u_2
    inst✝² : PseudoMetricSpace β
    inst✝¹ : Monoid β
    inst✝ : LipschitzMul β
    ⊢ LipschitzWith (LipschitzMul.C β) fun p => HMul.hMul p.1 p.2
  -/
  exact lipschitzWith_lipschitz_const_mul_edist
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

@[to_additive]
instance (priority := 100) LipschitzMul.continuousMul : ContinuousMul β :=
  ⟨lipschitzWith_lipschitz_const_mul_edist.continuous⟩


@[to_additive]
instance Submonoid.lipschitzMul (s : Submonoid β) : LipschitzMul s where
  lipschitz_mul := ⟨LipschitzMul.C β, by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      s : Submonoid β
      ⊢ LipschitzWith (LipschitzMul.C β) fun p => HMul.hMul p.1 p.2
    -/
    rintro ⟨x₁, x₂⟩ ⟨y₁, y₂⟩
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      s : Submonoid β
      x₁ x₂ y₁ y₂ : Subtype fun x => Membership.mem s x
      ⊢ LE.le (EDist.edist ((fun p => HMul.hMul p.1 p.2) { fst := x₁, snd := x₂ }) ( …
    -/
    convert lipschitzWith_lipschitz_const_mul_edist ⟨(x₁ : β), x₂⟩ ⟨y₁, y₂⟩ using 1⟩
    /-
      🎉 no goals
    -/


@[to_additive]
instance MulOpposite.lipschitzMul : LipschitzMul βᵐᵒᵖ where
  lipschitz_mul := ⟨LipschitzMul.C β, fun ⟨x₁, x₂⟩ ⟨y₁, y₂⟩ =>
    (lipschitzWith_lipschitz_const_mul_edist ⟨x₂.unop, x₁.unop⟩ ⟨y₂.unop, y₁.unop⟩).trans_eq
      (congr_arg _ <| max_comm _ _)⟩

-- this instance could be deduced from `NormedAddCommGroup.lipschitzAdd`, but we prove it
-- separately here so that it is available earlier in the hierarchy

instance Real.hasLipschitzAdd : LipschitzAdd ℝ where
  lipschitz_add := ⟨2, LipschitzWith.of_dist_le_mul fun p q => by
    simp only [Real.dist_eq, Prod.dist_eq, Prod.fst_sub, Prod.snd_sub, NNReal.coe_ofNat,
      add_sub_add_comm, two_mul]
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      p q : Prod Real Real
      ⊢ LE.le (abs (HAdd.hAdd (HSub.hSub p.1 q.1) (HSub.hSub p.2 q.2))) (HAdd.hAdd ( …
    -/
    refine le_trans (abs_add (p.1 - q.1) (p.2 - q.2)) ?_
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      p q : Prod Real Real
      ⊢ LE.le (HAdd.hAdd (abs (HSub.hSub p.1 q.1)) (abs (HSub.hSub p.2 q.2))) (HAdd. …
    -/
    exact add_le_add (le_max_left _ _) (le_max_right _ _)⟩
    /-
      🎉 no goals
    -/

-- this instance has the same proof as `AddSubmonoid.lipschitzAdd`, but the former can't
-- directly be applied here since `ℝ≥0` is a subtype of `ℝ`, not an additive submonoid.

instance NNReal.hasLipschitzAdd : LipschitzAdd ℝ≥0 where
  lipschitz_add := ⟨LipschitzAdd.C ℝ, by
    /-
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      ⊢ LipschitzWith (LipschitzAdd.C Real) fun p => HAdd.hAdd p.1 p.2
    -/
    rintro ⟨x₁, x₂⟩ ⟨y₁, y₂⟩
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      inst✝³ : PseudoMetricSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : Monoid β
      inst✝ : LipschitzMul β
      x₁ x₂ y₁ y₂ : NNReal
      ⊢ LE.le (EDist.edist ((fun p => HAdd.hAdd p.1 p.2) { fst := x₁, snd := x₂ }) ( …
    -/
    exact lipschitzWith_lipschitz_const_add_edist ⟨(x₁ : ℝ), x₂⟩ ⟨y₁, y₂⟩⟩
    /-
      🎉 no goals
    -/


/-- Mixin typeclass on a scalar action of a metric space `α` on a metric space `β` both with
distinguished points `0`, requiring compatibility of the action in the sense that
`dist (x • y₁) (x • y₂) ≤ dist x 0 * dist y₁ y₂` and
`dist (x₁ • y) (x₂ • y) ≤ dist x₁ x₂ * dist y 0`. -/
class BoundedSMul : Prop where
  dist_smul_pair' : ∀ x : α, ∀ y₁ y₂ : β, dist (x • y₁) (x • y₂) ≤ dist x 0 * dist y₁ y₂
  dist_pair_smul' : ∀ x₁ x₂ : α, ∀ y : β, dist (x₁ • y) (x₂ • y) ≤ dist x₁ x₂ * dist y 0


theorem dist_smul_pair (x : α) (y₁ y₂ : β) : dist (x • y₁) (x • y₂) ≤ dist x 0 * dist y₁ y₂ :=
  BoundedSMul.dist_smul_pair' x y₁ y₂


theorem dist_pair_smul (x₁ x₂ : α) (y : β) : dist (x₁ • y) (x₂ • y) ≤ dist x₁ x₂ * dist y 0 :=
  BoundedSMul.dist_pair_smul' x₁ x₂ y

-- see Note [lower instance priority]

/-- The typeclass `BoundedSMul` on a metric-space scalar action implies continuity of the action. -/
instance (priority := 100) BoundedSMul.continuousSMul : ContinuousSMul α β where
  continuous_smul := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : BoundedSMul α β
      ⊢ Continuous fun p => HSMul.hSMul p.1 p.2
    -/
    rw [Metric.continuous_iff]
    /-
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : BoundedSMul α β
      ⊢ ∀ (b : Prod α β) (ε : Real), GT.gt ε 0 → Exists fun δ => And (GT.gt δ 0) (∀  …
    -/
    rintro ⟨a, b⟩ ε ε0
    obtain ⟨δ, δ0, hδε⟩ : ∃ δ > 0, δ * (δ + dist b 0) + dist a 0 * δ < ε := by
      have : Continuous fun δ ↦ δ * (δ + dist b 0) + dist a 0 * δ := by fun_prop
      refine ((this.tendsto' _ _ ?_).eventually (gt_mem_nhds ε0)).exists_gt
      simp
    /-
      case mk.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : BoundedSMul α β
      a : α
      b : β
      ε : Real
      ε0 : GT.gt ε 0
      δ : Real
      δ0 : GT.gt δ 0
      hδε : LT.lt (HAdd.hAdd (HMul.hMul δ (HAdd.hAdd δ (Dist.dist b 0))) (HMul.hMul  …
      ⊢ Exists fun δ => And (GT.gt δ 0) (∀ (a_1 : Prod α β), LT.lt (Dist.dist a_1 {  …
    -/
    refine ⟨δ, δ0, fun (a', b') hab' => ?_⟩
    /-
      case mk.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝⁵ : PseudoMetricSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero α
      inst✝² : Zero β
      inst✝¹ : SMul α β
      inst✝ : BoundedSMul α β
      a : α
      b : β
      ε : Real
      ε0 : GT.gt ε 0
      δ : Real
      δ0 : GT.gt δ 0
      hδε : LT.lt (HAdd.hAdd (HMul.hMul δ (HAdd.hAdd δ (Dist.dist b 0))) (HMul.hMul  …
      x✝ : Prod α β
      a' : α
      b' : β
      hab' : LT.lt (Dist.dist { fst := a', snd := b' } { fst := a, snd := b }) δ
      ⊢ LT.lt (Dist.dist (HSMul.hSMul { fst := a', snd := b' }.1 { fst := a', snd := …
    -/
    obtain ⟨ha, hb⟩ := max_lt_iff.1 hab'
    calc dist (a' • b') (a • b)
        ≤ dist (a' • b') (a • b') + dist (a • b') (a • b) := dist_triangle ..
      _ ≤ dist a' a * dist b' 0 + dist a 0 * dist b' b :=
        add_le_add (dist_pair_smul _ _ _) (dist_smul_pair _ _ _)
      _ ≤ δ * (δ + dist b 0) + dist a 0 * δ := by
          have : dist b' 0 ≤ δ + dist b 0 := (dist_triangle _ _ _).trans <| add_le_add_right hb.le _
          gcongr
      _ < ε := hδε


instance (priority := 100) BoundedSMul.toUniformContinuousConstSMul :
    UniformContinuousConstSMul α β :=
  ⟨fun c => ((lipschitzWith_iff_dist_le_mul (K := nndist c 0)).2 fun _ _ =>
    dist_smul_pair c _ _).uniformContinuous⟩

-- this instance could be deduced from `NormedSpace.boundedSMul`, but we prove it separately
-- here so that it is available earlier in the hierarchy

instance Real.boundedSMul : BoundedSMul ℝ ℝ where
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝⁵ : PseudoMetricSpace α
                                  inst✝⁴ : PseudoMetricSpace β
                                  inst✝³ : Zero α
                                  inst✝² : Zero β
                                  inst✝¹ : SMul α β
                                  inst✝ : BoundedSMul α β
                                  x y₁ y₂ : Real
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x y₁) (HSMul.hSMul x y₂)) (HMul.hMul (Dist.dis …
                                -/
  dist_smul_pair' x y₁ y₂ := by simpa [Real.dist_eq, mul_sub] using (abs_mul x (y₁ - y₂)).le
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝⁵ : PseudoMetricSpace α
                                  inst✝⁴ : PseudoMetricSpace β
                                  inst✝³ : Zero α
                                  inst✝² : Zero β
                                  inst✝¹ : SMul α β
                                  inst✝ : BoundedSMul α β
                                  x₁ x₂ y : Real
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x₁ y) (HSMul.hSMul x₂ y)) (HMul.hMul (Dist.dis …
                                -/
  dist_pair_smul' x₁ x₂ y := by simpa [Real.dist_eq, sub_mul] using (abs_mul (x₁ - x₂) y).le
                                /-
                                  🎉 no goals
                                -/


instance NNReal.boundedSMul : BoundedSMul ℝ≥0 ℝ≥0 where
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝⁵ : PseudoMetricSpace α
                                  inst✝⁴ : PseudoMetricSpace β
                                  inst✝³ : Zero α
                                  inst✝² : Zero β
                                  inst✝¹ : SMul α β
                                  inst✝ : BoundedSMul α β
                                  x y₁ y₂ : NNReal
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x y₁) (HSMul.hSMul x y₂)) (HMul.hMul (Dist.dis …
                                -/
  dist_smul_pair' x y₁ y₂ := by convert dist_smul_pair (x : ℝ) (y₁ : ℝ) y₂ using 1
                                /-
                                  🎉 no goals
                                -/
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  inst✝⁵ : PseudoMetricSpace α
                                  inst✝⁴ : PseudoMetricSpace β
                                  inst✝³ : Zero α
                                  inst✝² : Zero β
                                  inst✝¹ : SMul α β
                                  inst✝ : BoundedSMul α β
                                  x₁ x₂ y : NNReal
                                  ⊢ LE.le (Dist.dist (HSMul.hSMul x₁ y) (HSMul.hSMul x₂ y)) (HMul.hMul (Dist.dis …
                                -/
  dist_pair_smul' x₁ x₂ y := by convert dist_pair_smul (x₁ : ℝ) x₂ (y : ℝ) using 1
                                /-
                                  🎉 no goals
                                -/


/-- If a scalar is central, then its right action is bounded when its left action is. -/
instance BoundedSMul.op [SMul αᵐᵒᵖ β] [IsCentralScalar α β] : BoundedSMul αᵐᵒᵖ β where
  dist_smul_pair' :=
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         inst✝⁷ : PseudoMetricSpace α
                                         inst✝⁶ : PseudoMetricSpace β
                                         inst✝⁵ : Zero α
                                         inst✝⁴ : Zero β
                                         inst✝³ : SMul α β
                                         inst✝² : BoundedSMul α β
                                         inst✝¹ : SMul (MulOpposite α) β
                                         inst✝ : IsCentralScalar α β
                                         x : α
                                         y₁ y₂ : β
                                         ⊢ LE.le (Dist.dist (HSMul.hSMul (MulOpposite.op x) y₁) (HSMul.hSMul (MulOpposi …
                                       -/
    MulOpposite.rec' fun x y₁ y₂ => by simpa only [op_smul_eq_smul] using dist_smul_pair x y₁ y₂
                                       /-
                                         🎉 no goals
                                       -/
  dist_pair_smul' :=
    MulOpposite.rec' fun x₁ =>
                                      /-
                                        α : Type u_1
                                        β : Type u_2
                                        inst✝⁷ : PseudoMetricSpace α
                                        inst✝⁶ : PseudoMetricSpace β
                                        inst✝⁵ : Zero α
                                        inst✝⁴ : Zero β
                                        inst✝³ : SMul α β
                                        inst✝² : BoundedSMul α β
                                        inst✝¹ : SMul (MulOpposite α) β
                                        inst✝ : IsCentralScalar α β
                                        x₁ x₂ : α
                                        y : β
                                        ⊢ LE.le (Dist.dist (HSMul.hSMul (MulOpposite.op x₁) y) (HSMul.hSMul (MulOpposi …
                                      -/
      MulOpposite.rec' fun x₂ y => by simpa only [op_smul_eq_smul] using dist_pair_smul x₁ x₂ y
                                      /-
                                        🎉 no goals
                                      -/


instance [Monoid α] [LipschitzMul α] : LipschitzAdd (Additive α) :=
  ⟨@LipschitzMul.lipschitz_mul α _ _ _⟩


instance [AddMonoid α] [LipschitzAdd α] : LipschitzMul (Multiplicative α) :=
  ⟨@LipschitzAdd.lipschitz_add α _ _ _⟩


@[to_additive]
instance [Monoid α] [LipschitzMul α] : LipschitzMul αᵒᵈ :=
  ‹LipschitzMul α›


instance Pi.instBoundedSMul {α : Type*} {β : ι → Type*} [PseudoMetricSpace α]
    [∀ i, PseudoMetricSpace (β i)] [Zero α] [∀ i, Zero (β i)] [∀ i, SMul α (β i)]
    [∀ i, BoundedSMul α (β i)] : BoundedSMul α (∀ i, β i) where
  dist_smul_pair' x y₁ y₂ :=
                          /-
                            α✝ : Type u_1
                            β✝ : Type u_2
                            inst✝⁸ : PseudoMetricSpace α✝
                            inst✝⁷ : PseudoMetricSpace β✝
                            ι : Type u_3
                            inst✝⁶ : Fintype ι
                            α : Type u_4
                            β : ι → Type u_5
                            inst✝⁵ : PseudoMetricSpace α
                            inst✝⁴ : (i : ι) → PseudoMetricSpace (β i)
                            inst✝³ : Zero α
                            inst✝² : (i : ι) → Zero (β i)
                            inst✝¹ : (i : ι) → SMul α (β i)
                            inst✝ : ∀ (i : ι), BoundedSMul α (β i)
                            x : α
                            y₁ y₂ : (i : ι) → β i
                            ⊢ LE.le 0 (HMul.hMul (Dist.dist x 0) (Dist.dist y₁ y₂))
                          -/
    (dist_pi_le_iff <| by positivity).2 fun _ ↦
                          /-
                            🎉 no goals
                          -/
      (dist_smul_pair _ _ _).trans <| mul_le_mul_of_nonneg_left (dist_le_pi_dist _ _ _) dist_nonneg
  dist_pair_smul' x₁ x₂ y :=
                          /-
                            α✝ : Type u_1
                            β✝ : Type u_2
                            inst✝⁸ : PseudoMetricSpace α✝
                            inst✝⁷ : PseudoMetricSpace β✝
                            ι : Type u_3
                            inst✝⁶ : Fintype ι
                            α : Type u_4
                            β : ι → Type u_5
                            inst✝⁵ : PseudoMetricSpace α
                            inst✝⁴ : (i : ι) → PseudoMetricSpace (β i)
                            inst✝³ : Zero α
                            inst✝² : (i : ι) → Zero (β i)
                            inst✝¹ : (i : ι) → SMul α (β i)
                            inst✝ : ∀ (i : ι), BoundedSMul α (β i)
                            x₁ x₂ : α
                            y : (i : ι) → β i
                            ⊢ LE.le 0 (HMul.hMul (Dist.dist x₁ x₂) (Dist.dist y 0))
                          -/
    (dist_pi_le_iff <| by positivity).2 fun _ ↦
                          /-
                            🎉 no goals
                          -/
      (dist_pair_smul _ _ _).trans <| mul_le_mul_of_nonneg_left (dist_le_pi_dist _ 0 _) dist_nonneg


instance Pi.instBoundedSMul' {α β : ι → Type*} [∀ i, PseudoMetricSpace (α i)]
    [∀ i, PseudoMetricSpace (β i)] [∀ i, Zero (α i)] [∀ i, Zero (β i)] [∀ i, SMul (α i) (β i)]
    [∀ i, BoundedSMul (α i) (β i)] : BoundedSMul (∀ i, α i) (∀ i, β i) where
  dist_smul_pair' x y₁ y₂ :=
                          /-
                            α✝ : Type u_1
                            β✝ : Type u_2
                            inst✝⁸ : PseudoMetricSpace α✝
                            inst✝⁷ : PseudoMetricSpace β✝
                            ι : Type u_3
                            inst✝⁶ : Fintype ι
                            α : ι → Type u_4
                            β : ι → Type u_5
                            inst✝⁵ : (i : ι) → PseudoMetricSpace (α i)
                            inst✝⁴ : (i : ι) → PseudoMetricSpace (β i)
                            inst✝³ : (i : ι) → Zero (α i)
                            inst✝² : (i : ι) → Zero (β i)
                            inst✝¹ : (i : ι) → SMul (α i) (β i)
                            inst✝ : ∀ (i : ι), BoundedSMul (α i) (β i)
                            x : (i : ι) → α i
                            y₁ y₂ : (i : ι) → β i
                            ⊢ LE.le 0 (HMul.hMul (Dist.dist x 0) (Dist.dist y₁ y₂))
                          -/
    (dist_pi_le_iff <| by positivity).2 fun _ ↦
                          /-
                            🎉 no goals
                          -/
      (dist_smul_pair _ _ _).trans <|
        mul_le_mul (dist_le_pi_dist _ 0 _) (dist_le_pi_dist _ _ _) dist_nonneg dist_nonneg
  dist_pair_smul' x₁ x₂ y :=
                          /-
                            α✝ : Type u_1
                            β✝ : Type u_2
                            inst✝⁸ : PseudoMetricSpace α✝
                            inst✝⁷ : PseudoMetricSpace β✝
                            ι : Type u_3
                            inst✝⁶ : Fintype ι
                            α : ι → Type u_4
                            β : ι → Type u_5
                            inst✝⁵ : (i : ι) → PseudoMetricSpace (α i)
                            inst✝⁴ : (i : ι) → PseudoMetricSpace (β i)
                            inst✝³ : (i : ι) → Zero (α i)
                            inst✝² : (i : ι) → Zero (β i)
                            inst✝¹ : (i : ι) → SMul (α i) (β i)
                            inst✝ : ∀ (i : ι), BoundedSMul (α i) (β i)
                            x₁ x₂ : (i : ι) → α i
                            y : (i : ι) → β i
                            ⊢ LE.le 0 (HMul.hMul (Dist.dist x₁ x₂) (Dist.dist y 0))
                          -/
    (dist_pi_le_iff <| by positivity).2 fun _ ↦
                          /-
                            🎉 no goals
                          -/
      (dist_pair_smul _ _ _).trans <|
        mul_le_mul (dist_le_pi_dist _ _ _) (dist_le_pi_dist _ 0 _) dist_nonneg dist_nonneg


instance Prod.instBoundedSMul {α β γ : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β]
    [PseudoMetricSpace γ] [Zero α] [Zero β] [Zero γ] [SMul α β] [SMul α γ] [BoundedSMul α β]
    [BoundedSMul α γ] : BoundedSMul α (β × γ) where
  dist_smul_pair' _x _y₁ _y₂ :=
    max_le ((dist_smul_pair _ _ _).trans <| mul_le_mul_of_nonneg_left (le_max_left _ _) dist_nonneg)
      ((dist_smul_pair _ _ _).trans <| mul_le_mul_of_nonneg_left (le_max_right _ _) dist_nonneg)
  dist_pair_smul' _x₁ _x₂ _y :=
    max_le ((dist_pair_smul _ _ _).trans <| mul_le_mul_of_nonneg_left (le_max_left _ _) dist_nonneg)
      ((dist_pair_smul _ _ _).trans <| mul_le_mul_of_nonneg_left (le_max_right _ _) dist_nonneg)


instance {α β : Type*}
    [PseudoMetricSpace α] [PseudoMetricSpace β] [Zero α] [Zero β] [SMul α β] [BoundedSMul α β] :
    BoundedSMul α (SeparationQuotient β) where
  dist_smul_pair' _ := Quotient.ind₂ <| dist_smul_pair _
  dist_pair_smul' _ _ := Quotient.ind <| dist_pair_smul _ _

-- We don't have the `SMul α γ → SMul β δ → SMul (α × β) (γ × δ)` instance, but if we did, then
-- `BoundedSMul α γ → BoundedSMul β δ → BoundedSMul (α × β) (γ × δ)` would hold

