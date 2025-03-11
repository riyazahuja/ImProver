/-- A copy of a Pi type, on which we will put the `L^p` distance. Since the Pi type itself is
already endowed with the `L^∞` distance, we need the type synonym to avoid confusing typeclass
resolution. Also, we let it depend on `p`, to get a whole family of type on which we can put
different distances. -/
abbrev PiLp (p : ℝ≥0∞) {ι : Type*} (α : ι → Type*) : Type _ :=
  WithLp p (∀ i : ι, α i)

/-The following should not be a `FunLike` instance because then the coercion `⇑` would get
unfolded to `FunLike.coe` instead of `WithLp.equiv`. -/

instance (p : ℝ≥0∞) {ι : Type*} (α : ι → Type*) : CoeFun (PiLp p α) (fun _ ↦ (i : ι) → α i) where
  coe := WithLp.equiv p _


instance (p : ℝ≥0∞) {ι : Type*} (α : ι → Type*) [∀ i, Inhabited (α i)] : Inhabited (PiLp p α) :=
  ⟨fun _ => default⟩


@[ext]
protected theorem PiLp.ext {p : ℝ≥0∞} {ι : Type*} {α : ι → Type*} {x y : PiLp p α}
    (h : ∀ i, x i = y i) : x = y := funext h


@[simp, nolint simpNF]
theorem zero_apply : (0 : PiLp p β) i = 0 :=
  rfl


@[simp]
theorem add_apply : (x + y) i = x i + y i :=
  rfl


@[simp]
theorem sub_apply : (x - y) i = x i - y i :=
  rfl


@[simp]
theorem smul_apply : (c • x) i = c • x i :=
  rfl


@[simp]
theorem neg_apply : (-x) i = -x i :=
  rfl


variable (p) in
/-- The projection on the `i`-th coordinate of `WithLp p (∀ i, α i)`, as a linear map. -/
@[simps!]
def projₗ (i : ι) : PiLp p β →ₗ[𝕜] β i :=
  (LinearMap.proj i : (∀ i, β i) →ₗ[𝕜] β i) ∘ₗ (WithLp.linearEquiv p 𝕜 (∀ i, β i)).toLinearMap


@[simp]
theorem _root_.WithLp.equiv_pi_apply (x : PiLp p α) (i : ι) : WithLp.equiv p _ x i = x i :=
  rfl


@[simp]
theorem  _root_.WithLp.equiv_symm_pi_apply (x : ∀ i, α i) (i : ι) :
    (WithLp.equiv p _).symm x i = x i :=
  rfl


/-- Endowing the space `PiLp p β` with the `L^p` edistance. We register this instance
separate from `pi_Lp.pseudo_emetric` since the latter requires the type class hypothesis
`[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future emetric-like structure on `PiLp p β` for `p < 1`
satisfying a relaxed triangle inequality. The terminology for this varies throughout the
literature, but it is sometimes called a *quasi-metric* or *semi-metric*. -/
instance : EDist (PiLp p β) where
  edist f g :=
    if p = 0 then {i | edist (f i) (g i) ≠ 0}.toFinite.toFinset.card
    else
      if p = ∞ then ⨆ i, edist (f i) (g i) else (∑ i, edist (f i) (g i) ^ p.toReal) ^ (1 / p.toReal)


theorem edist_eq_card (f g : PiLp 0 β) :
    edist f g = {i | edist (f i) (g i) ≠ 0}.toFinite.toFinset.card :=
  if_pos rfl


theorem edist_eq_sum {p : ℝ≥0∞} (hp : 0 < p.toReal) (f g : PiLp p β) :
    edist f g = (∑ i, edist (f i) (g i) ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


theorem edist_eq_iSup (f g : PiLp ∞ β) : edist f g = ⨆ i, edist (f i) (g i) := rfl


/-- This holds independent of `p` and does not require `[Fact (1 ≤ p)]`. We keep it separate
from `pi_Lp.pseudo_emetric_space` so it can be used also for `p < 1`. -/
protected theorem edist_self (f : PiLp p β) : edist f f = 0 := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → PseudoEMetricSpace (β i)
    f : PiLp p β
    ⊢ Eq (EDist.edist f f) 0
  -/
  rcases p.trichotomy with (rfl | rfl | h)
    /-
      case inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f : PiLp 0 β
      ⊢ Eq (EDist.edist f f) 0
    -/
  · simp [edist_eq_card]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f : PiLp Top.top β
      ⊢ Eq (EDist.edist f f) 0
    -/
  · simp [edist_eq_iSup]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f : PiLp p β
      h : LT.lt 0 p.toReal
      ⊢ Eq (EDist.edist f f) 0
    -/
  · simp [edist_eq_sum h, ENNReal.zero_rpow_of_pos h, ENNReal.zero_rpow_of_pos (inv_pos.2 <| h)]
    /-
      🎉 no goals
    -/


/-- This holds independent of `p` and does not require `[Fact (1 ≤ p)]`. We keep it separate
from `pi_Lp.pseudo_emetric_space` so it can be used also for `p < 1`. -/
protected theorem edist_comm (f g : PiLp p β) : edist f g = edist g f := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → PseudoEMetricSpace (β i)
    f g : PiLp p β
    ⊢ Eq (EDist.edist f g) (EDist.edist g f)
  -/
  rcases p.trichotomy with (rfl | rfl | h)
    /-
      case inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f g : PiLp 0 β
      ⊢ Eq (EDist.edist f g) (EDist.edist g f)
    -/
  · simp only [edist_eq_card, edist_comm]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f g : PiLp Top.top β
      ⊢ Eq (EDist.edist f g) (EDist.edist g f)
    -/
  · simp only [edist_eq_iSup, edist_comm]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → PseudoEMetricSpace (β i)
      f g : PiLp p β
      h : LT.lt 0 p.toReal
      ⊢ Eq (EDist.edist f g) (EDist.edist g f)
    -/
  · simp only [edist_eq_sum h, edist_comm]
    /-
      🎉 no goals
    -/


/-- Endowing the space `PiLp p β` with the `L^p` distance. We register this instance
separate from `pi_Lp.pseudo_metric` since the latter requires the type class hypothesis
`[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future metric-like structure on `PiLp p β` for `p < 1`
satisfying a relaxed triangle inequality. The terminology for this varies throughout the
literature, but it is sometimes called a *quasi-metric* or *semi-metric*. -/
instance : Dist (PiLp p α) where
  dist f g :=
    if p = 0 then {i | dist (f i) (g i) ≠ 0}.toFinite.toFinset.card
    else
      if p = ∞ then ⨆ i, dist (f i) (g i) else (∑ i, dist (f i) (g i) ^ p.toReal) ^ (1 / p.toReal)


theorem dist_eq_card (f g : PiLp 0 α) :
    dist f g = {i | dist (f i) (g i) ≠ 0}.toFinite.toFinset.card :=
  if_pos rfl


theorem dist_eq_sum {p : ℝ≥0∞} (hp : 0 < p.toReal) (f g : PiLp p α) :
    dist f g = (∑ i, dist (f i) (g i) ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


theorem dist_eq_iSup (f g : PiLp ∞ α) : dist f g = ⨆ i, dist (f i) (g i) := rfl


/-- Endowing the space `PiLp p β` with the `L^p` norm. We register this instance
separate from `PiLp.seminormedAddCommGroup` since the latter requires the type class hypothesis
`[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future norm-like structure on `PiLp p β` for `p < 1`
satisfying a relaxed triangle inequality. These are called *quasi-norms*. -/
instance instNorm : Norm (PiLp p β) where
  norm f :=
    if p = 0 then {i | ‖f i‖ ≠ 0}.toFinite.toFinset.card
    else if p = ∞ then ⨆ i, ‖f i‖ else (∑ i, ‖f i‖ ^ p.toReal) ^ (1 / p.toReal)


theorem norm_eq_card (f : PiLp 0 β) : ‖f‖ = {i | ‖f i‖ ≠ 0}.toFinite.toFinset.card :=
  if_pos rfl


theorem norm_eq_ciSup (f : PiLp ∞ β) : ‖f‖ = ⨆ i, ‖f i‖ := rfl


theorem norm_eq_sum (hp : 0 < p.toReal) (f : PiLp p β) :
    ‖f‖ = (∑ i, ‖f i‖ ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


/-- Endowing the space `PiLp p β` with the `L^p` pseudoemetric structure. This definition is not
satisfactory, as it does not register the fact that the topology and the uniform structure coincide
with the product one. Therefore, we do not register it as an instance. Using this as a temporary
pseudoemetric space instance, we will show that the uniform structure is equal (but not defeq) to
the product one, and then register an instance in which we replace the uniform structure by the
product one using this pseudoemetric space and `PseudoEMetricSpace.replaceUniformity`. -/
def pseudoEmetricAux : PseudoEMetricSpace (PiLp p β) where
  edist_self := PiLp.edist_self p
  edist_comm := PiLp.edist_comm p
  edist_triangle f g h := by
    /-
      p : ENNReal
      𝕜 : Type u_1
      ι : Type u_2
      α : ι → Type u_3
      β : ι → Type u_4
      inst✝³ : Fact (LE.le 1 p)
      inst✝² : (i : ι) → PseudoMetricSpace (α i)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      f g h : PiLp p β
      ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
    -/
    rcases p.dichotomy with (rfl | hp)
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : (i : ι) → PseudoMetricSpace (α i)
        inst✝² : (i : ι) → PseudoEMetricSpace (β i)
        inst✝¹ : Fintype ι
        inst✝ : Fact (LE.le 1 Top.top)
        f g h : PiLp Top.top β
        ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
      -/
    · simp only [edist_eq_iSup]
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : (i : ι) → PseudoMetricSpace (α i)
        inst✝² : (i : ι) → PseudoEMetricSpace (β i)
        inst✝¹ : Fintype ι
        inst✝ : Fact (LE.le 1 Top.top)
        f g h : PiLp Top.top β
        ⊢ LE.le (iSup fun i => EDist.edist (f i) (h i)) (HAdd.hAdd (iSup fun i => EDis …
      -/
      cases isEmpty_or_nonempty ι
        /-
          case inl.inl
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝³ : (i : ι) → PseudoMetricSpace (α i)
          inst✝² : (i : ι) → PseudoEMetricSpace (β i)
          inst✝¹ : Fintype ι
          inst✝ : Fact (LE.le 1 Top.top)
          f g h : PiLp Top.top β
          h✝ : IsEmpty ι
          ⊢ LE.le (iSup fun i => EDist.edist (f i) (h i)) (HAdd.hAdd (iSup fun i => EDis …
        -/
      · simp only [ciSup_of_empty, ENNReal.bot_eq_zero, add_zero, nonpos_iff_eq_zero]
        /-
          🎉 no goals
        -/
      -- Porting note: `le_iSup` needed some help
      refine
        iSup_le fun i => (edist_triangle _ (g i) _).trans <| add_le_add
            (le_iSup (fun k => edist (f k) (g k)) i) (le_iSup (fun k => edist (g k) (h k)) i)
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : Fact (LE.le 1 p)
        inst✝² : (i : ι) → PseudoMetricSpace (α i)
        inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
        inst✝ : Fintype ι
        f g h : PiLp p β
        hp : LE.le 1 p.toReal
        ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
      -/
    · simp only [edist_eq_sum (zero_lt_one.trans_le hp)]
      calc
        (∑ i, edist (f i) (h i) ^ p.toReal) ^ (1 / p.toReal) ≤
            (∑ i, (edist (f i) (g i) + edist (g i) (h i)) ^ p.toReal) ^ (1 / p.toReal) := by
          gcongr
          apply edist_triangle
        _ ≤
            (∑ i, edist (f i) (g i) ^ p.toReal) ^ (1 / p.toReal) +
              (∑ i, edist (g i) (h i) ^ p.toReal) ^ (1 / p.toReal) :=
          ENNReal.Lp_add_le _ _ _ hp


/-- An auxiliary lemma used twice in the proof of `PiLp.pseudoMetricAux` below. Not intended for
use outside this file. -/
theorem iSup_edist_ne_top_aux {ι : Type*} [Finite ι] {α : ι → Type*}
    [∀ i, PseudoMetricSpace (α i)] (f g : PiLp ∞ α) : (⨆ i, edist (f i) (g i)) ≠ ⊤ := by
  /-
    ι : Type u_5
    inst✝¹ : Finite ι
    α : ι → Type u_6
    inst✝ : (i : ι) → PseudoMetricSpace (α i)
    f g : PiLp Top.top α
    ⊢ Ne (iSup fun i => EDist.edist (f i) (g i)) Top.top
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_5
    inst✝¹ : Finite ι
    α : ι → Type u_6
    inst✝ : (i : ι) → PseudoMetricSpace (α i)
    f g : PiLp Top.top α
    val✝ : Fintype ι
    ⊢ Ne (iSup fun i => EDist.edist (f i) (g i)) Top.top
  -/
  obtain ⟨M, hM⟩ := Finite.exists_le fun i => (⟨dist (f i) (g i), dist_nonneg⟩ : ℝ≥0)
  /-
    case intro.intro
    ι : Type u_5
    inst✝¹ : Finite ι
    α : ι → Type u_6
    inst✝ : (i : ι) → PseudoMetricSpace (α i)
    f g : PiLp Top.top α
    val✝ : Fintype ι
    M : Subtype fun r => LE.le 0 r
    hM : ∀ (i : ι), LE.le ⟨Dist.dist (f i) (g i), ⋯⟩ M
    ⊢ Ne (iSup fun i => EDist.edist (f i) (g i)) Top.top
  -/
  refine ne_of_lt ((iSup_le fun i => ?_).trans_lt (@ENNReal.coe_lt_top M))
  /-
    case intro.intro
    ι : Type u_5
    inst✝¹ : Finite ι
    α : ι → Type u_6
    inst✝ : (i : ι) → PseudoMetricSpace (α i)
    f g : PiLp Top.top α
    val✝ : Fintype ι
    M : Subtype fun r => LE.le 0 r
    hM : ∀ (i : ι), LE.le ⟨Dist.dist (f i) (g i), ⋯⟩ M
    i : ι
    ⊢ LE.le (EDist.edist (f i) (g i)) ↑M
  -/
  simp only [edist, PseudoMetricSpace.edist_dist, ENNReal.ofReal_eq_coe_nnreal dist_nonneg]
  /-
    case intro.intro
    ι : Type u_5
    inst✝¹ : Finite ι
    α : ι → Type u_6
    inst✝ : (i : ι) → PseudoMetricSpace (α i)
    f g : PiLp Top.top α
    val✝ : Fintype ι
    M : Subtype fun r => LE.le 0 r
    hM : ∀ (i : ι), LE.le ⟨Dist.dist (f i) (g i), ⋯⟩ M
    i : ι
    ⊢ LE.le ↑⟨Dist.dist (f i) (g i), ⋯⟩ ↑M
  -/
  exact mod_cast hM i
  /-
    🎉 no goals
  -/


/-- Endowing the space `PiLp p α` with the `L^p` pseudometric structure. This definition is not
satisfactory, as it does not register the fact that the topology, the uniform structure, and the
bornology coincide with the product ones. Therefore, we do not register it as an instance. Using
this as a temporary pseudoemetric space instance, we will show that the uniform structure is equal
(but not defeq) to the product one, and then register an instance in which we replace the uniform
structure and the bornology by the product ones using this pseudometric space,
`PseudoMetricSpace.replaceUniformity`, and `PseudoMetricSpace.replaceBornology`.

See note [reducible non-instances] -/
abbrev pseudoMetricAux : PseudoMetricSpace (PiLp p α) :=
  PseudoEMetricSpace.toPseudoMetricSpaceOfDist dist
    (fun f g => by
      /-
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : Fact (LE.le 1 p)
        inst✝² : (i : ι) → PseudoMetricSpace (α i)
        inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
        inst✝ : Fintype ι
        f g : PiLp p α
        ⊢ Ne (EDist.edist f g) Top.top
      -/
      rcases p.dichotomy with (rfl | h)
        /-
          case inl
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝³ : (i : ι) → PseudoMetricSpace (α i)
          inst✝² : (i : ι) → PseudoEMetricSpace (β i)
          inst✝¹ : Fintype ι
          inst✝ : Fact (LE.le 1 Top.top)
          f g : PiLp Top.top α
          ⊢ Ne (EDist.edist f g) Top.top
        -/
      · exact iSup_edist_ne_top_aux f g
        /-
          🎉 no goals
        -/
        /-
          case inr
          p : ENNReal
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝³ : Fact (LE.le 1 p)
          inst✝² : (i : ι) → PseudoMetricSpace (α i)
          inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
          inst✝ : Fintype ι
          f g : PiLp p α
          h : LE.le 1 p.toReal
          ⊢ Ne (EDist.edist f g) Top.top
        -/
      · rw [edist_eq_sum (zero_lt_one.trans_le h)]
        exact ENNReal.rpow_ne_top_of_nonneg (by positivity) <| ENNReal.sum_ne_top.2 fun _ _ ↦
          ENNReal.rpow_ne_top_of_nonneg (by positivity) (edist_ne_top _ _))
    fun f g => by
    /-
      p : ENNReal
      𝕜 : Type u_1
      ι : Type u_2
      α : ι → Type u_3
      β : ι → Type u_4
      inst✝³ : Fact (LE.le 1 p)
      inst✝² : (i : ι) → PseudoMetricSpace (α i)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      f g : PiLp p α
      ⊢ Eq (Dist.dist f g) (EDist.edist f g).toReal
    -/
    rcases p.dichotomy with (rfl | h)
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : (i : ι) → PseudoMetricSpace (α i)
        inst✝² : (i : ι) → PseudoEMetricSpace (β i)
        inst✝¹ : Fintype ι
        inst✝ : Fact (LE.le 1 Top.top)
        f g : PiLp Top.top α
        ⊢ Eq (Dist.dist f g) (EDist.edist f g).toReal
      -/
    · rw [edist_eq_iSup, dist_eq_iSup]
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝³ : (i : ι) → PseudoMetricSpace (α i)
        inst✝² : (i : ι) → PseudoEMetricSpace (β i)
        inst✝¹ : Fintype ι
        inst✝ : Fact (LE.le 1 Top.top)
        f g : PiLp Top.top α
        ⊢ Eq (iSup fun i => Dist.dist (f i) (g i)) (iSup fun i => EDist.edist (f i) (g …
      -/
      cases isEmpty_or_nonempty ι
        /-
          case inl.inl
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝³ : (i : ι) → PseudoMetricSpace (α i)
          inst✝² : (i : ι) → PseudoEMetricSpace (β i)
          inst✝¹ : Fintype ι
          inst✝ : Fact (LE.le 1 Top.top)
          f g : PiLp Top.top α
          h✝ : IsEmpty ι
          ⊢ Eq (iSup fun i => Dist.dist (f i) (g i)) (iSup fun i => EDist.edist (f i) (g …
        -/
      · simp only [Real.iSup_of_isEmpty, ciSup_of_empty, ENNReal.bot_eq_zero, ENNReal.zero_toReal]
        /-
          🎉 no goals
        -/
        /-
          case inl.inr
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝³ : (i : ι) → PseudoMetricSpace (α i)
          inst✝² : (i : ι) → PseudoEMetricSpace (β i)
          inst✝¹ : Fintype ι
          inst✝ : Fact (LE.le 1 Top.top)
          f g : PiLp Top.top α
          h✝ : Nonempty ι
          ⊢ Eq (iSup fun i => Dist.dist (f i) (g i)) (iSup fun i => EDist.edist (f i) (g …
        -/
      · refine le_antisymm (ciSup_le fun i => ?_) ?_
        · rw [← ENNReal.ofReal_le_iff_le_toReal (iSup_edist_ne_top_aux f g), ←
            PseudoMetricSpace.edist_dist]
          -- Porting note: `le_iSup` needed some help
          /-
            case inl.inr.refine_1
            𝕜 : Type u_1
            ι : Type u_2
            α : ι → Type u_3
            β : ι → Type u_4
            inst✝³ : (i : ι) → PseudoMetricSpace (α i)
            inst✝² : (i : ι) → PseudoEMetricSpace (β i)
            inst✝¹ : Fintype ι
            inst✝ : Fact (LE.le 1 Top.top)
            f g : PiLp Top.top α
            h✝ : Nonempty ι
            i : ι
            ⊢ LE.le (PseudoMetricSpace.edist (f i) (g i)) (iSup fun i => EDist.edist (f i) …
          -/
          exact le_iSup (fun k => edist (f k) (g k)) i
          /-
            🎉 no goals
          -/
          /-
            case inl.inr.refine_2
            𝕜 : Type u_1
            ι : Type u_2
            α : ι → Type u_3
            β : ι → Type u_4
            inst✝³ : (i : ι) → PseudoMetricSpace (α i)
            inst✝² : (i : ι) → PseudoEMetricSpace (β i)
            inst✝¹ : Fintype ι
            inst✝ : Fact (LE.le 1 Top.top)
            f g : PiLp Top.top α
            h✝ : Nonempty ι
            ⊢ LE.le (iSup fun i => EDist.edist (f i) (g i)).toReal (iSup fun i => Dist.dis …
          -/
        · refine ENNReal.toReal_le_of_le_ofReal (Real.sSup_nonneg ?_) (iSup_le fun i => ?_)
            /-
              case inl.inr.refine_2.refine_1
              𝕜 : Type u_1
              ι : Type u_2
              α : ι → Type u_3
              β : ι → Type u_4
              inst✝³ : (i : ι) → PseudoMetricSpace (α i)
              inst✝² : (i : ι) → PseudoEMetricSpace (β i)
              inst✝¹ : Fintype ι
              inst✝ : Fact (LE.le 1 Top.top)
              f g : PiLp Top.top α
              h✝ : Nonempty ι
              ⊢ ∀ (x : Real), Membership.mem (Set.range fun i => Dist.dist (f i) (g i)) x →  …
            -/
          · rintro - ⟨i, rfl⟩
            /-
              case inl.inr.refine_2.refine_1.intro
              𝕜 : Type u_1
              ι : Type u_2
              α : ι → Type u_3
              β : ι → Type u_4
              inst✝³ : (i : ι) → PseudoMetricSpace (α i)
              inst✝² : (i : ι) → PseudoEMetricSpace (β i)
              inst✝¹ : Fintype ι
              inst✝ : Fact (LE.le 1 Top.top)
              f g : PiLp Top.top α
              h✝ : Nonempty ι
              i : ι
              ⊢ LE.le 0 ((fun i => Dist.dist (f i) (g i)) i)
            -/
            exact dist_nonneg
            /-
              🎉 no goals
            -/
            /-
              case inl.inr.refine_2.refine_2
              𝕜 : Type u_1
              ι : Type u_2
              α : ι → Type u_3
              β : ι → Type u_4
              inst✝³ : (i : ι) → PseudoMetricSpace (α i)
              inst✝² : (i : ι) → PseudoEMetricSpace (β i)
              inst✝¹ : Fintype ι
              inst✝ : Fact (LE.le 1 Top.top)
              f g : PiLp Top.top α
              h✝ : Nonempty ι
              i : ι
              ⊢ LE.le (EDist.edist (f i) (g i)) (ENNReal.ofReal (iSup fun i => Dist.dist (f  …
            -/
          · change PseudoMetricSpace.edist _ _ ≤ _
            /-
              case inl.inr.refine_2.refine_2
              𝕜 : Type u_1
              ι : Type u_2
              α : ι → Type u_3
              β : ι → Type u_4
              inst✝³ : (i : ι) → PseudoMetricSpace (α i)
              inst✝² : (i : ι) → PseudoEMetricSpace (β i)
              inst✝¹ : Fintype ι
              inst✝ : Fact (LE.le 1 Top.top)
              f g : PiLp Top.top α
              h✝ : Nonempty ι
              i : ι
              ⊢ LE.le (PseudoMetricSpace.edist (f i) (g i)) (ENNReal.ofReal (iSup fun i => D …
            -/
            rw [PseudoMetricSpace.edist_dist]
            -- Porting note: `le_ciSup` needed some help
            exact ENNReal.ofReal_le_ofReal
              (le_ciSup (Finite.bddAbove_range (fun k => dist (f k) (g k))) i)
    · have A : ∀ i, edist (f i) (g i) ^ p.toReal ≠ ⊤ := fun i =>
        ENNReal.rpow_ne_top_of_nonneg (zero_le_one.trans h) (edist_ne_top _ _)
      simp only [edist_eq_sum (zero_lt_one.trans_le h), dist_edist, ENNReal.toReal_rpow,
        dist_eq_sum (zero_lt_one.trans_le h), ← ENNReal.toReal_sum fun i _ => A i]


theorem lipschitzWith_equiv_aux : LipschitzWith 1 (WithLp.equiv p (∀ i, β i)) := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
    inst✝ : Fintype ι
    ⊢ LipschitzWith 1 ⇑(WithLp.equiv p ((i : ι) → β i))
  -/
  intro x y
  simp_rw [ENNReal.coe_one, one_mul, edist_pi_def, Finset.sup_le_iff, Finset.mem_univ,
    forall_true_left, WithLp.equiv_pi_apply]
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
    inst✝ : Fintype ι
    x y : WithLp p ((i : ι) → β i)
    ⊢ ∀ (b : ι), LE.le (EDist.edist (x b) (y b)) (EDist.edist x y)
  -/
  rcases p.dichotomy with (rfl | h)
    /-
      case inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : (i : ι) → PseudoEMetricSpace (β i)
      inst✝¹ : Fintype ι
      inst✝ : Fact (LE.le 1 Top.top)
      x y : WithLp Top.top ((i : ι) → β i)
      ⊢ ∀ (b : ι), LE.le (EDist.edist (x b) (y b)) (EDist.edist x y)
    -/
  · simpa only [edist_eq_iSup] using le_iSup fun i => edist (x i) (y i)
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      ⊢ ∀ (b : ι), LE.le (EDist.edist (x b) (y b)) (EDist.edist x y)
    -/
  · have cancel : p.toReal * (1 / p.toReal) = 1 := mul_div_cancel₀ 1 (zero_lt_one.trans_le h).ne'
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ ∀ (b : ι), LE.le (EDist.edist (x b) (y b)) (EDist.edist x y)
    -/
    rw [edist_eq_sum (zero_lt_one.trans_le h)]
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ ∀ (b : ι), LE.le (EDist.edist (x b) (y b)) (HPow.hPow (Finset.univ.sum fun i …
    -/
    intro i
    calc
      edist (x i) (y i) = (edist (x i) (y i) ^ p.toReal) ^ (1 / p.toReal) := by
        simp [← ENNReal.rpow_mul, cancel, -one_div]
      _ ≤ (∑ i, edist (x i) (y i) ^ p.toReal) ^ (1 / p.toReal) := by
        gcongr
        exact Finset.single_le_sum (fun i _ => (bot_le : (0 : ℝ≥0∞) ≤ _)) (Finset.mem_univ i)


theorem antilipschitzWith_equiv_aux :
    AntilipschitzWith ((Fintype.card ι : ℝ≥0) ^ (1 / p).toReal) (WithLp.equiv p (∀ i, β i)) := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
    inst✝ : Fintype ι
    ⊢ AntilipschitzWith (HPow.hPow (↑(Fintype.card ι)) (HDiv.hDiv 1 p).toReal) ⇑(W …
  -/
  intro x y
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
    inst✝ : Fintype ι
    x y : WithLp p ((i : ι) → β i)
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow (↑(Fintype.card ι)) (HDiv.hD …
  -/
  rcases p.dichotomy with (rfl | h)
  · simp only [edist_eq_iSup, ENNReal.div_top, ENNReal.zero_toReal, NNReal.rpow_zero,
      ENNReal.coe_one, one_mul, iSup_le_iff]
    -- Porting note: `Finset.le_sup` needed some help
    /-
      case inl
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : (i : ι) → PseudoEMetricSpace (β i)
      inst✝¹ : Fintype ι
      inst✝ : Fact (LE.le 1 Top.top)
      x y : WithLp Top.top ((i : ι) → β i)
      ⊢ ∀ (i : ι), LE.le (EDist.edist (x i) (y i)) (EDist.edist ((WithLp.equiv Top.t …
    -/
    exact fun i => Finset.le_sup (f := fun i => edist (x i) (y i)) (Finset.mem_univ i)
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow (↑(Fintype.card ι)) (HDiv.hD …
    -/
  · have pos : 0 < p.toReal := zero_lt_one.trans_le h
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow (↑(Fintype.card ι)) (HDiv.hD …
    -/
    have nonneg : 0 ≤ 1 / p.toReal := one_div_nonneg.2 (le_of_lt pos)
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow (↑(Fintype.card ι)) (HDiv.hD …
    -/
    have cancel : p.toReal * (1 / p.toReal) = 1 := mul_div_cancel₀ 1 (ne_of_gt pos)
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow (↑(Fintype.card ι)) (HDiv.hD …
    -/
    rw [edist_eq_sum pos, ENNReal.toReal_div 1 p]
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      β : ι → Type u_4
      inst✝² : Fact (LE.le 1 p)
      inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
      inst✝ : Fintype ι
      x y : WithLp p ((i : ι) → β i)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (EDist.edist (x i) (y i …
    -/
    simp only [edist, ← one_div, ENNReal.one_toReal]
    calc
      (∑ i, edist (x i) (y i) ^ p.toReal) ^ (1 / p.toReal) ≤
          (∑ _i, edist (WithLp.equiv p _ x) (WithLp.equiv p _ y) ^ p.toReal) ^ (1 / p.toReal) := by
        gcongr with i
        exact Finset.le_sup (f := fun i => edist (x i) (y i)) (Finset.mem_univ i)
      _ =
          ((Fintype.card ι : ℝ≥0) ^ (1 / p.toReal) : ℝ≥0) *
            edist (WithLp.equiv p _ x) (WithLp.equiv p _ y) := by
        simp only [nsmul_eq_mul, Finset.card_univ, ENNReal.rpow_one, Finset.sum_const,
          ENNReal.mul_rpow_of_nonneg _ _ nonneg, ← ENNReal.rpow_mul, cancel]
        have : (Fintype.card ι : ℝ≥0∞) = (Fintype.card ι : ℝ≥0) :=
          (ENNReal.coe_natCast (Fintype.card ι)).symm
        rw [this, ENNReal.coe_rpow_of_nonneg _ nonneg]


theorem aux_uniformity_eq : 𝓤 (PiLp p β) = 𝓤[Pi.uniformSpace _] := by
  have A : IsUniformInducing (WithLp.equiv p (∀ i, β i)) :=
    (antilipschitzWith_equiv_aux p β).isUniformInducing
      (lipschitzWith_equiv_aux p β).uniformContinuous
  have : (fun x : PiLp p β × PiLp p β => (WithLp.equiv p _ x.fst, WithLp.equiv p _ x.snd)) = id :=
    by ext i <;> rfl
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    inst✝² : Fact (LE.le 1 p)
    inst✝¹ : (i : ι) → PseudoEMetricSpace (β i)
    inst✝ : Fintype ι
    A : IsUniformInducing ⇑(WithLp.equiv p ((i : ι) → β i))
    this : Eq (fun x => { fst := (WithLp.equiv p ((i : ι) → β i)) x.1, snd := (Wit …
    ⊢ Eq (uniformity (PiLp p β)) (uniformity ((i : ι) → β i))
  -/
  rw [← A.comap_uniformity, this, comap_id]
  /-
    🎉 no goals
  -/


theorem aux_cobounded_eq : cobounded (PiLp p α) = @cobounded _ Pi.instBornology :=
  calc
    cobounded (PiLp p α) = comap (WithLp.equiv p (∀ i, α i)) (cobounded _) :=
      le_antisymm (antilipschitzWith_equiv_aux p α).tendsto_cobounded.le_comap
        (lipschitzWith_equiv_aux p α).comap_cobounded_le
    _ = _ := comap_id


instance uniformSpace [∀ i, UniformSpace (β i)] : UniformSpace (PiLp p β) :=
  Pi.uniformSpace _


theorem uniformContinuous_equiv [∀ i, UniformSpace (β i)] :
    UniformContinuous (WithLp.equiv p (∀ i, β i)) :=
  uniformContinuous_id


theorem uniformContinuous_equiv_symm [∀ i, UniformSpace (β i)] :
    UniformContinuous (WithLp.equiv p (∀ i, β i)).symm :=
  uniformContinuous_id


@[continuity]
theorem continuous_equiv [∀ i, UniformSpace (β i)] : Continuous (WithLp.equiv p (∀ i, β i)) :=
  continuous_id


@[continuity]
theorem continuous_equiv_symm [∀ i, UniformSpace (β i)] :
    Continuous (WithLp.equiv p (∀ i, β i)).symm :=
  continuous_id


instance bornology [∀ i, Bornology (β i)] : Bornology (PiLp p β) :=
  Pi.instBornology



/-- pseudoemetric space instance on the product of finitely many pseudoemetric spaces, using the
`L^p` pseudoedistance, and having as uniformity the product uniformity. -/
instance [∀ i, PseudoEMetricSpace (β i)] : PseudoEMetricSpace (PiLp p β) :=
  (pseudoEmetricAux p β).replaceUniformity (aux_uniformity_eq p β).symm


/-- emetric space instance on the product of finitely many emetric spaces, using the `L^p`
edistance, and having as uniformity the product uniformity. -/
instance [∀ i, EMetricSpace (α i)] : EMetricSpace (PiLp p α) :=
  @EMetricSpace.ofT0PseudoEMetricSpace (PiLp p α) _ Pi.instT0Space


/-- pseudometric space instance on the product of finitely many pseudometric spaces, using the
`L^p` distance, and having as uniformity the product uniformity. -/
instance [∀ i, PseudoMetricSpace (β i)] : PseudoMetricSpace (PiLp p β) :=
  ((pseudoMetricAux p β).replaceUniformity (aux_uniformity_eq p β).symm).replaceBornology fun s =>
    Filter.ext_iff.1 (aux_cobounded_eq p β).symm sᶜ


/-- metric space instance on the product of finitely many metric spaces, using the `L^p` distance,
and having as uniformity the product uniformity. -/
instance [∀ i, MetricSpace (α i)] : MetricSpace (PiLp p α) :=
  MetricSpace.ofT0PseudoMetricSpace _


theorem nndist_eq_sum {p : ℝ≥0∞} [Fact (1 ≤ p)] {β : ι → Type*} [∀ i, PseudoMetricSpace (β i)]
    (hp : p ≠ ∞) (x y : PiLp p β) :
    nndist x y = (∑ i : ι, nndist (x i) (y i) ^ p.toReal) ^ (1 / p.toReal) :=
  -- Porting note: was `Subtype.ext`
  NNReal.eq <| by
    /-
      ι : Type u_2
      inst✝² : Fintype ι
      p : ENNReal
      inst✝¹ : Fact (LE.le 1 p)
      β : ι → Type u_5
      inst✝ : (i : ι) → PseudoMetricSpace (β i)
      hp : Ne p Top.top
      x y : PiLp p β
      ⊢ Eq ↑(NNDist.nndist x y) ↑(HPow.hPow (Finset.univ.sum fun i => HPow.hPow (NND …
    -/
    push_cast
    /-
      ι : Type u_2
      inst✝² : Fintype ι
      p : ENNReal
      inst✝¹ : Fact (LE.le 1 p)
      β : ι → Type u_5
      inst✝ : (i : ι) → PseudoMetricSpace (β i)
      hp : Ne p Top.top
      x y : PiLp p β
      ⊢ Eq (Dist.dist x y) (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow (Dist.di …
    -/
    exact dist_eq_sum (p.toReal_pos_iff_ne_top.mpr hp) _ _
    /-
      🎉 no goals
    -/


theorem nndist_eq_iSup {β : ι → Type*} [∀ i, PseudoMetricSpace (β i)] (x y : PiLp ∞ β) :
    nndist x y = ⨆ i, nndist (x i) (y i) :=
  -- Porting note: was `Subtype.ext`
  NNReal.eq <| by
    /-
      ι : Type u_2
      inst✝¹ : Fintype ι
      β : ι → Type u_5
      inst✝ : (i : ι) → PseudoMetricSpace (β i)
      x y : PiLp Top.top β
      ⊢ Eq ↑(NNDist.nndist x y) ↑(iSup fun i => NNDist.nndist (x i) (y i))
    -/
    push_cast
    /-
      ι : Type u_2
      inst✝¹ : Fintype ι
      β : ι → Type u_5
      inst✝ : (i : ι) → PseudoMetricSpace (β i)
      x y : PiLp Top.top β
      ⊢ Eq (Dist.dist x y) (iSup fun i => Dist.dist (x i) (y i))
    -/
    exact dist_eq_iSup _ _
    /-
      🎉 no goals
    -/


theorem lipschitzWith_equiv [∀ i, PseudoEMetricSpace (β i)] :
    LipschitzWith 1 (WithLp.equiv p (∀ i, β i)) :=
  lipschitzWith_equiv_aux p β


theorem antilipschitzWith_equiv [∀ i, PseudoEMetricSpace (β i)] :
    AntilipschitzWith ((Fintype.card ι : ℝ≥0) ^ (1 / p).toReal) (WithLp.equiv p (∀ i, β i)) :=
  antilipschitzWith_equiv_aux p β


theorem infty_equiv_isometry [∀ i, PseudoEMetricSpace (β i)] :
    Isometry (WithLp.equiv ∞ (∀ i, β i)) :=
  fun x y =>
                  /-
                    ι : Type u_2
                    β : ι → Type u_4
                    inst✝¹ : Fintype ι
                    inst✝ : (i : ι) → PseudoEMetricSpace (β i)
                    x y : WithLp Top.top ((i : ι) → β i)
                    ⊢ LE.le (EDist.edist ((WithLp.equiv Top.top ((i : ι) → β i)) x) ((WithLp.equiv …
                  -/
  le_antisymm (by simpa only [ENNReal.coe_one, one_mul] using lipschitzWith_equiv ∞ β x y)
                  /-
                    🎉 no goals
                  -/
    (by
      simpa only [ENNReal.div_top, ENNReal.zero_toReal, NNReal.rpow_zero, ENNReal.coe_one,
        one_mul] using antilipschitzWith_equiv ∞ β x y)


/-- seminormed group instance on the product of finitely many normed groups, using the `L^p`
norm. -/
instance seminormedAddCommGroup [∀ i, SeminormedAddCommGroup (β i)] :
    SeminormedAddCommGroup (PiLp p β) :=
  { Pi.addCommGroup with
    dist_eq := fun x y => by
      /-
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝¹ : Fintype ι
        inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
        x y : PiLp p β
        ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
      -/
      rcases p.dichotomy with (rfl | h)
        /-
          case inl
          𝕜 : Type u_1
          ι : Type u_2
          α : ι → Type u_3
          β : ι → Type u_4
          inst✝¹ : Fintype ι
          inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
          hp : Fact (LE.le 1 Top.top)
          x y : PiLp Top.top β
          ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
        -/
      · simp only [dist_eq_iSup, norm_eq_ciSup, dist_eq_norm, sub_apply]
        /-
          🎉 no goals
        -/
      · have : p ≠ ∞ := by
          intro hp
          rw [hp, ENNReal.top_toReal] at h
          linarith
        simp only [dist_eq_sum (zero_lt_one.trans_le h), norm_eq_sum (zero_lt_one.trans_le h),
          dist_eq_norm, sub_apply] }


/-- normed group instance on the product of finitely many normed groups, using the `L^p` norm. -/
instance normedAddCommGroup [∀ i, NormedAddCommGroup (α i)] : NormedAddCommGroup (PiLp p α) :=
  { PiLp.seminormedAddCommGroup p α with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


theorem nnnorm_eq_sum {p : ℝ≥0∞} [Fact (1 ≤ p)] {β : ι → Type*} (hp : p ≠ ∞)
    [∀ i, SeminormedAddCommGroup (β i)] (f : PiLp p β) :
    ‖f‖₊ = (∑ i, ‖f i‖₊ ^ p.toReal) ^ (1 / p.toReal) := by
  /-
    ι : Type u_2
    inst✝² : Fintype ι
    p : ENNReal
    inst✝¹ : Fact (LE.le 1 p)
    β : ι → Type u_5
    hp : Ne p Top.top
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp p β
    ⊢ Eq (NNNorm.nnnorm f) (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (NNNorm. …
  -/
  ext
  /-
    case a
    ι : Type u_2
    inst✝² : Fintype ι
    p : ENNReal
    inst✝¹ : Fact (LE.le 1 p)
    β : ι → Type u_5
    hp : Ne p Top.top
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp p β
    ⊢ Eq ↑(NNNorm.nnnorm f) ↑(HPow.hPow (Finset.univ.sum fun i => HPow.hPow (NNNor …
  -/
  simp [NNReal.coe_sum, norm_eq_sum (p.toReal_pos_iff_ne_top.mpr hp)]
  /-
    🎉 no goals
  -/


theorem nnnorm_eq_ciSup (f : PiLp ∞ β) : ‖f‖₊ = ⨆ i, ‖f i‖₊ := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp Top.top β
    ⊢ Eq (NNNorm.nnnorm f) (iSup fun i => NNNorm.nnnorm (f i))
  -/
  ext
  /-
    case a
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp Top.top β
    ⊢ Eq ↑(NNNorm.nnnorm f) ↑(iSup fun i => NNNorm.nnnorm (f i))
  -/
  simp [NNReal.coe_iSup, norm_eq_ciSup]
  /-
    🎉 no goals
  -/


@[simp] theorem nnnorm_equiv (f : PiLp ∞ β) : ‖WithLp.equiv ⊤ _ f‖₊ = ‖f‖₊ := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp Top.top β
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv Top.top ((i : ι) → β i)) f)) (NNNorm.nnnorm …
  -/
  rw [nnnorm_eq_ciSup, Pi.nnnorm_def, Finset.sup_univ_eq_ciSup]
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    f : PiLp Top.top β
    ⊢ Eq (iSup fun i => NNNorm.nnnorm ((WithLp.equiv Top.top ((i : ι) → β i)) f i) …
  -/
  dsimp only [WithLp.equiv_pi_apply]
  /-
    🎉 no goals
  -/


@[simp] theorem nnnorm_equiv_symm (f : ∀ i, β i) : ‖(WithLp.equiv ⊤ _).symm f‖₊ = ‖f‖₊ :=
  (nnnorm_equiv _).symm


@[simp] theorem norm_equiv (f : PiLp ∞ β) : ‖WithLp.equiv ⊤ _ f‖ = ‖f‖ :=
  congr_arg NNReal.toReal <| nnnorm_equiv f


@[simp] theorem norm_equiv_symm (f : ∀ i, β i) : ‖(WithLp.equiv ⊤ _).symm f‖ = ‖f‖ :=
  (norm_equiv _).symm


theorem norm_eq_of_nat {p : ℝ≥0∞} [Fact (1 ≤ p)] {β : ι → Type*}
    [∀ i, SeminormedAddCommGroup (β i)] (n : ℕ) (h : p = n) (f : PiLp p β) :
    ‖f‖ = (∑ i, ‖f i‖ ^ n) ^ (1 / (n : ℝ)) := by
  /-
    ι : Type u_2
    inst✝² : Fintype ι
    p : ENNReal
    inst✝¹ : Fact (LE.le 1 p)
    β : ι → Type u_5
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    n : Nat
    h : Eq p ↑n
    f : PiLp p β
    ⊢ Eq (Norm.norm f) (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (Norm.norm ( …
  -/
  have := p.toReal_pos_iff_ne_top.mpr (ne_of_eq_of_ne h <| ENNReal.natCast_ne_top n)
  simp only [one_div, h, Real.rpow_natCast, ENNReal.toReal_nat, eq_self_iff_true, Finset.sum_congr,
    norm_eq_sum this]


theorem norm_eq_of_L1 (x : PiLp 1 β) : ‖x‖ = ∑ i : ι, ‖x i‖ := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x : PiLp 1 β
    ⊢ Eq (Norm.norm x) (Finset.univ.sum fun i => Norm.norm (x i))
  -/
  simp [norm_eq_sum]
  /-
    🎉 no goals
  -/


theorem nnnorm_eq_of_L1 (x : PiLp 1 β) : ‖x‖₊ = ∑ i : ι, ‖x i‖₊ :=
                  /-
                    ι : Type u_2
                    β : ι → Type u_4
                    inst✝¹ : Fintype ι
                    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
                    x : PiLp 1 β
                    ⊢ Eq ↑(NNNorm.nnnorm x) ↑(Finset.univ.sum fun i => NNNorm.nnnorm (x i))
                  -/
  NNReal.eq <| by push_cast; exact norm_eq_of_L1 x
                             /-
                               🎉 no goals
                             -/


theorem dist_eq_of_L1 (x y : PiLp 1 β) : dist x y = ∑ i, dist (x i) (y i) := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x y : PiLp 1 β
    ⊢ Eq (Dist.dist x y) (Finset.univ.sum fun i => Dist.dist (x i) (y i))
  -/
  simp_rw [dist_eq_norm, norm_eq_of_L1, sub_apply]
  /-
    🎉 no goals
  -/


theorem nndist_eq_of_L1 (x y : PiLp 1 β) : nndist x y = ∑ i, nndist (x i) (y i) :=
                  /-
                    ι : Type u_2
                    β : ι → Type u_4
                    inst✝¹ : Fintype ι
                    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
                    x y : PiLp 1 β
                    ⊢ Eq ↑(NNDist.nndist x y) ↑(Finset.univ.sum fun i => NNDist.nndist (x i) (y i))
                  -/
  NNReal.eq <| by push_cast; exact dist_eq_of_L1 _ _
                             /-
                               🎉 no goals
                             -/


theorem edist_eq_of_L1 (x y : PiLp 1 β) : edist x y = ∑ i, edist (x i) (y i) := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x y : PiLp 1 β
    ⊢ Eq (EDist.edist x y) (Finset.univ.sum fun i => EDist.edist (x i) (y i))
  -/
  simp [PiLp.edist_eq_sum]
  /-
    🎉 no goals
  -/


theorem norm_eq_of_L2 (x : PiLp 2 β) :
    ‖x‖ = √(∑ i : ι, ‖x i‖ ^ 2) := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x : PiLp 2 β
    ⊢ Eq (Norm.norm x) (Finset.univ.sum fun i => HPow.hPow (Norm.norm (x i)) 2).sqrt
  -/
  rw [norm_eq_of_nat 2 (by norm_cast) _]
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x : PiLp 2 β
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (Norm.norm (x i)) 2) (HDiv …
  -/
  rw [Real.sqrt_eq_rpow]
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x : PiLp 2 β
    ⊢ Eq (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (Norm.norm (x i)) 2) (HDiv …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem nnnorm_eq_of_L2 (x : PiLp 2 β) :
    ‖x‖₊ = NNReal.sqrt (∑ i : ι, ‖x i‖₊ ^ 2) :=
  NNReal.eq <| by
    /-
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
      x : PiLp 2 β
      ⊢ Eq ↑(NNNorm.nnnorm x) ↑(NNReal.sqrt (Finset.univ.sum fun i => HPow.hPow (NNN …
    -/
    push_cast
    /-
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
      x : PiLp 2 β
      ⊢ Eq (Norm.norm x) (Finset.univ.sum fun x_1 => HPow.hPow (Norm.norm (x x_1)) 2 …
    -/
    exact norm_eq_of_L2 x
    /-
      🎉 no goals
    -/


theorem norm_sq_eq_of_L2 (β : ι → Type*) [∀ i, SeminormedAddCommGroup (β i)] (x : PiLp 2 β) :
    ‖x‖ ^ 2 = ∑ i : ι, ‖x i‖ ^ 2 := by
  suffices ‖x‖₊ ^ 2 = ∑ i : ι, ‖x i‖₊ ^ 2 by
    simpa only [NNReal.coe_sum] using congr_arg ((↑) : ℝ≥0 → ℝ) this
  /-
    ι : Type u_2
    inst✝¹ : Fintype ι
    β : ι → Type u_5
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x : PiLp 2 β
    ⊢ Eq (HPow.hPow (NNNorm.nnnorm x) 2) (Finset.univ.sum fun i => HPow.hPow (NNNo …
  -/
  rw [nnnorm_eq_of_L2, NNReal.sq_sqrt]
  /-
    🎉 no goals
  -/


theorem dist_eq_of_L2 (x y : PiLp 2 β) :
    dist x y = √(∑ i, dist (x i) (y i) ^ 2) := by
  /-
    ι : Type u_2
    β : ι → Type u_4
    inst✝¹ : Fintype ι
    inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
    x y : PiLp 2 β
    ⊢ Eq (Dist.dist x y) (Finset.univ.sum fun i => HPow.hPow (Dist.dist (x i) (y i …
  -/
  simp_rw [dist_eq_norm, norm_eq_of_L2, sub_apply]
  /-
    🎉 no goals
  -/


theorem nndist_eq_of_L2 (x y : PiLp 2 β) :
    nndist x y = NNReal.sqrt (∑ i, nndist (x i) (y i) ^ 2) :=
  NNReal.eq <| by
    /-
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
      x y : PiLp 2 β
      ⊢ Eq ↑(NNDist.nndist x y) ↑(NNReal.sqrt (Finset.univ.sum fun i => HPow.hPow (N …
    -/
    push_cast
    /-
      ι : Type u_2
      β : ι → Type u_4
      inst✝¹ : Fintype ι
      inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
      x y : PiLp 2 β
      ⊢ Eq (Dist.dist x y) (Finset.univ.sum fun x_1 => HPow.hPow (Dist.dist (x x_1)  …
    -/
    exact dist_eq_of_L2 _ _
    /-
      🎉 no goals
    -/


theorem edist_eq_of_L2 (x y : PiLp 2 β) :
                                                                 /-
                                                                   ι : Type u_2
                                                                   β : ι → Type u_4
                                                                   inst✝¹ : Fintype ι
                                                                   inst✝ : (i : ι) → SeminormedAddCommGroup (β i)
                                                                   x y : PiLp 2 β
                                                                   ⊢ Eq (EDist.edist x y) (HPow.hPow (Finset.univ.sum fun i => HPow.hPow (EDist.e …
                                                                 -/
    edist x y = (∑ i, edist (x i) (y i) ^ 2) ^ (1 / 2 : ℝ) := by simp [PiLp.edist_eq_sum]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


instance instBoundedSMul [SeminormedRing 𝕜] [∀ i, SeminormedAddCommGroup (β i)]
    [∀ i, Module 𝕜 (β i)] [∀ i, BoundedSMul 𝕜 (β i)] :
    BoundedSMul 𝕜 (PiLp p β) :=
  .of_nnnorm_smul_le fun c f => by
    /-
      p : ENNReal
      𝕜 : Type u_1
      ι : Type u_2
      α : ι → Type u_3
      β : ι → Type u_4
      hp : Fact (LE.le 1 p)
      inst✝⁴ : Fintype ι
      inst✝³ : SeminormedRing 𝕜
      inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
      inst✝¹ : (i : ι) → Module 𝕜 (β i)
      inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
      c : 𝕜
      f : PiLp p β
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
    -/
    rcases p.dichotomy with (rfl | hp)
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        hp : Fact (LE.le 1 Top.top)
        f : PiLp Top.top β
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
    · rw [← nnnorm_equiv, ← nnnorm_equiv, WithLp.equiv_smul]
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        hp : Fact (LE.le 1 Top.top)
        f : PiLp Top.top β
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c ((WithLp.equiv Top.top ((i : ι) → β i))  …
      -/
      exact nnnorm_smul_le c (WithLp.equiv ∞ (∀ i, β i) f)
      /-
        🎉 no goals
      -/
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp✝ : Fact (LE.le 1 p)
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        f : PiLp p β
        hp : LE.le 1 p.toReal
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
    · have hp0 : 0 < p.toReal := zero_lt_one.trans_le hp
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp✝ : Fact (LE.le 1 p)
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        f : PiLp p β
        hp : LE.le 1 p.toReal
        hp0 : LT.lt 0 p.toReal
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
      have hpt : p ≠ ⊤ := p.toReal_pos_iff_ne_top.mp hp0
      rw [nnnorm_eq_sum hpt, nnnorm_eq_sum hpt, one_div, NNReal.rpow_inv_le_iff hp0,
        NNReal.mul_rpow, ← NNReal.rpow_mul, inv_mul_cancel₀ hp0.ne', NNReal.rpow_one,
        Finset.mul_sum]
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp✝ : Fact (LE.le 1 p)
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        f : PiLp p β
        hp : LE.le 1 p.toReal
        hp0 : LT.lt 0 p.toReal
        hpt : Ne p Top.top
        ⊢ LE.le (Finset.univ.sum fun i => HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c f i) …
      -/
      simp_rw [← NNReal.mul_rpow, smul_apply]
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp✝ : Fact (LE.le 1 p)
        inst✝⁴ : Fintype ι
        inst✝³ : SeminormedRing 𝕜
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (β i)
        inst✝ : ∀ (i : ι), BoundedSMul 𝕜 (β i)
        c : 𝕜
        f : PiLp p β
        hp : LE.le 1 p.toReal
        hp0 : LT.lt 0 p.toReal
        hpt : Ne p Top.top
        ⊢ LE.le (Finset.univ.sum fun x => HPow.hPow (NNNorm.nnnorm (HSMul.hSMul c (f x …
      -/
      exact Finset.sum_le_sum fun i _ => NNReal.rpow_le_rpow (nnnorm_smul_le _ _) hp0.le
      /-
        🎉 no goals
      -/


/-- The product of finitely many normed spaces is a normed space, with the `L^p` norm. -/
instance normedSpace [NormedField 𝕜] [∀ i, SeminormedAddCommGroup (β i)]
    [∀ i, NormedSpace 𝕜 (β i)] : NormedSpace 𝕜 (PiLp p β) where
  norm_smul_le := norm_smul_le


/-- The canonical map `WithLp.equiv` between `PiLp ∞ β` and `Π i, β i` as a linear isometric
equivalence. -/
def equivₗᵢ : PiLp ∞ β ≃ₗᵢ[𝕜] ∀ i, β i :=
  { WithLp.equiv ∞ (∀ i, β i) with
    map_add' := fun _f _g => rfl
    map_smul' := fun _c _f => rfl
    norm_map' := norm_equiv }


/-- An equivalence of finite domains induces a linearly isometric equivalence of finitely supported
functions -/
def _root_.LinearIsometryEquiv.piLpCongrLeft (e : ι ≃ ι') :
    (PiLp p fun _ : ι => E) ≃ₗᵢ[𝕜] PiLp p fun _ : ι' => E where
  toLinearEquiv := LinearEquiv.piCongrLeft' 𝕜 (fun _ : ι => E) e
  norm_map' x' := by
    /-
      p : ENNReal
      𝕜 : Type u_1
      ι : Type u_2
      α : ι → Type u_3
      β : ι → Type u_4
      hp : Fact (LE.le 1 p)
      inst✝⁸ : Fintype ι
      inst✝⁷ : Semiring 𝕜
      inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
      inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
      inst✝⁴ : (i : ι) → Module 𝕜 (α i)
      inst✝³ : (i : ι) → Module 𝕜 (β i)
      c : 𝕜
      ι' : Type u_5
      inst✝² : Fintype ι'
      E : Type u_6
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : Module 𝕜 E
      e : Equiv ι ι'
      x' : PiLp p fun x => E
      ⊢ Eq (Norm.norm ((LinearEquiv.piCongrLeft' 𝕜 (fun x => E) e) x')) (Norm.norm x')
    -/
    rcases p.dichotomy with (rfl | h)
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝⁸ : Fintype ι
        inst✝⁷ : Semiring 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝⁴ : (i : ι) → Module 𝕜 (α i)
        inst✝³ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        ι' : Type u_5
        inst✝² : Fintype ι'
        E : Type u_6
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : Module 𝕜 E
        e : Equiv ι ι'
        hp : Fact (LE.le 1 Top.top)
        x' : PiLp Top.top fun x => E
        ⊢ Eq (Norm.norm ((LinearEquiv.piCongrLeft' 𝕜 (fun x => E) e) x')) (Norm.norm x')
      -/
    · simp_rw [norm_eq_ciSup]
      /-
        case inl
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        inst✝⁸ : Fintype ι
        inst✝⁷ : Semiring 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝⁴ : (i : ι) → Module 𝕜 (α i)
        inst✝³ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        ι' : Type u_5
        inst✝² : Fintype ι'
        E : Type u_6
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : Module 𝕜 E
        e : Equiv ι ι'
        hp : Fact (LE.le 1 Top.top)
        x' : PiLp Top.top fun x => E
        ⊢ Eq (iSup fun i => Norm.norm ((LinearEquiv.piCongrLeft' 𝕜 (fun x => E) e) x'  …
      -/
      exact e.symm.iSup_congr fun _ => rfl
      /-
        🎉 no goals
      -/
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝⁸ : Fintype ι
        inst✝⁷ : Semiring 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝⁴ : (i : ι) → Module 𝕜 (α i)
        inst✝³ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        ι' : Type u_5
        inst✝² : Fintype ι'
        E : Type u_6
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : Module 𝕜 E
        e : Equiv ι ι'
        x' : PiLp p fun x => E
        h : LE.le 1 p.toReal
        ⊢ Eq (Norm.norm ((LinearEquiv.piCongrLeft' 𝕜 (fun x => E) e) x')) (Norm.norm x')
      -/
    · simp only [norm_eq_sum (zero_lt_one.trans_le h)]
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝⁸ : Fintype ι
        inst✝⁷ : Semiring 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝⁴ : (i : ι) → Module 𝕜 (α i)
        inst✝³ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        ι' : Type u_5
        inst✝² : Fintype ι'
        E : Type u_6
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : Module 𝕜 E
        e : Equiv ι ι'
        x' : PiLp p fun x => E
        h : LE.le 1 p.toReal
        ⊢ Eq (HPow.hPow (Finset.univ.sum fun x => HPow.hPow (Norm.norm ((LinearEquiv.p …
      -/
      congr 1
      /-
        case inr.e_a
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝⁸ : Fintype ι
        inst✝⁷ : Semiring 𝕜
        inst✝⁶ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝⁵ : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝⁴ : (i : ι) → Module 𝕜 (α i)
        inst✝³ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        ι' : Type u_5
        inst✝² : Fintype ι'
        E : Type u_6
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : Module 𝕜 E
        e : Equiv ι ι'
        x' : PiLp p fun x => E
        h : LE.le 1 p.toReal
        ⊢ Eq (Finset.univ.sum fun x => HPow.hPow (Norm.norm ((LinearEquiv.piCongrLeft' …
      -/
      exact Fintype.sum_equiv e.symm _ _ fun _ => rfl
      /-
        🎉 no goals
      -/


@[simp]
theorem _root_.LinearIsometryEquiv.piLpCongrLeft_apply (e : ι ≃ ι') (v : PiLp p fun _ : ι => E) :
    LinearIsometryEquiv.piLpCongrLeft p 𝕜 E e v = Equiv.piCongrLeft' (fun _ : ι => E) e v :=
  rfl


@[simp]
theorem _root_.LinearIsometryEquiv.piLpCongrLeft_symm (e : ι ≃ ι') :
    (LinearIsometryEquiv.piLpCongrLeft p 𝕜 E e).symm =
      LinearIsometryEquiv.piLpCongrLeft p 𝕜 E e.symm :=
  LinearIsometryEquiv.ext fun z ↦ -- Porting note: was `rfl`
    congr_arg (Equiv.toFun · z) (Equiv.piCongrLeft'_symm _ _)


@[simp high]
theorem _root_.LinearIsometryEquiv.piLpCongrLeft_single [DecidableEq ι] [DecidableEq ι']
    (e : ι ≃ ι') (i : ι) (v : E) :
    LinearIsometryEquiv.piLpCongrLeft p 𝕜 E e ((WithLp.equiv p (_ → E)).symm <| Pi.single i v) =
      (WithLp.equiv p (_ → E)).symm (Pi.single (e i) v) := by
  /-
    p : ENNReal
    𝕜 : Type u_1
    ι : Type u_2
    hp : Fact (LE.le 1 p)
    inst✝⁶ : Fintype ι
    inst✝⁵ : Semiring 𝕜
    ι' : Type u_5
    inst✝⁴ : Fintype ι'
    E : Type u_6
    inst✝³ : SeminormedAddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : DecidableEq ι
    inst✝ : DecidableEq ι'
    e : Equiv ι ι'
    i : ι
    v : E
    ⊢ Eq ((LinearIsometryEquiv.piLpCongrLeft p 𝕜 E e) ((WithLp.equiv p (ι → E)).sy …
  -/
  funext x
  simp [LinearIsometryEquiv.piLpCongrLeft_apply, LinearEquiv.piCongrLeft', Equiv.piCongrLeft',
    Pi.single, Function.update, Equiv.symm_apply_eq]


variable (p) in
/-- A family of linearly isometric equivalences in the codomain induces an isometric equivalence
between Pi types with the Lp norm.

This is the isometry version of `LinearEquiv.piCongrRight`. -/
protected def _root_.LinearIsometryEquiv.piLpCongrRight (e : ∀ i, α i ≃ₗᵢ[𝕜] β i) :
    PiLp p α ≃ₗᵢ[𝕜] PiLp p β where
  toLinearEquiv :=
    WithLp.linearEquiv _ _ _
      ≪≫ₗ (LinearEquiv.piCongrRight fun i => (e i).toLinearEquiv)
      ≪≫ₗ (WithLp.linearEquiv _ _ _).symm
  norm_map' := (WithLp.linearEquiv p 𝕜 _).symm.surjective.forall.2 fun x => by
    simp only [LinearEquiv.trans_apply, LinearEquiv.piCongrRight_apply,
      Equiv.apply_symm_apply, WithLp.linearEquiv_symm_apply, WithLp.linearEquiv_apply]
    /-
      p : ENNReal
      𝕜 : Type u_1
      ι : Type u_2
      α : ι → Type u_3
      β : ι → Type u_4
      hp : Fact (LE.le 1 p)
      inst✝⁵ : Fintype ι
      inst✝⁴ : Semiring 𝕜
      inst✝³ : (i : ι) → SeminormedAddCommGroup (α i)
      inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
      inst✝¹ : (i : ι) → Module 𝕜 (α i)
      inst✝ : (i : ι) → Module 𝕜 (β i)
      c : 𝕜
      e : (i : ι) → LinearIsometryEquiv (RingHom.id 𝕜) (α i) (β i)
      x : (i : ι) → α i
      ⊢ Eq (Norm.norm ((WithLp.equiv p ((i : ι) → β i)).symm ((LinearEquiv.piCongrRi …
    -/
    obtain rfl | hp := p.dichotomy
    · simp_rw [PiLp.norm_equiv_symm, Pi.norm_def, LinearEquiv.piCongrRight_apply,
        LinearIsometryEquiv.coe_toLinearEquiv, LinearIsometryEquiv.nnnorm_map]
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        ι : Type u_2
        α : ι → Type u_3
        β : ι → Type u_4
        hp✝ : Fact (LE.le 1 p)
        inst✝⁵ : Fintype ι
        inst✝⁴ : Semiring 𝕜
        inst✝³ : (i : ι) → SeminormedAddCommGroup (α i)
        inst✝² : (i : ι) → SeminormedAddCommGroup (β i)
        inst✝¹ : (i : ι) → Module 𝕜 (α i)
        inst✝ : (i : ι) → Module 𝕜 (β i)
        c : 𝕜
        e : (i : ι) → LinearIsometryEquiv (RingHom.id 𝕜) (α i) (β i)
        x : (i : ι) → α i
        hp : LE.le 1 p.toReal
        ⊢ Eq (Norm.norm ((WithLp.equiv p ((i : ι) → β i)).symm ((LinearEquiv.piCongrRi …
      -/
    · have : 0 < p.toReal := zero_lt_one.trans_le <| by norm_cast
      simp only [PiLp.norm_eq_sum this, WithLp.equiv_symm_pi_apply, LinearEquiv.piCongrRight_apply,
        LinearIsometryEquiv.coe_toLinearEquiv, LinearIsometryEquiv.norm_map]


@[simp]
theorem _root_.LinearIsometryEquiv.piLpCongrRight_apply (e : ∀ i, α i ≃ₗᵢ[𝕜] β i) (x : PiLp p α) :
    LinearIsometryEquiv.piLpCongrRight p e x =
      (WithLp.equiv p _).symm (fun i => e i (x i)) :=
  rfl


@[simp]
theorem _root_.LinearIsometryEquiv.piLpCongrRight_refl :
    LinearIsometryEquiv.piLpCongrRight p (fun i => .refl 𝕜 (α i)) = .refl _ _ :=
  rfl


@[simp]
theorem _root_.LinearIsometryEquiv.piLpCongrRight_symm (e : ∀ i, α i ≃ₗᵢ[𝕜] β i) :
    (LinearIsometryEquiv.piLpCongrRight p e).symm =
      LinearIsometryEquiv.piLpCongrRight p (fun i => (e i).symm) :=
  rfl


@[simp high]
theorem _root_.LinearIsometryEquiv.piLpCongrRight_single (e : ∀ i, α i ≃ₗᵢ[𝕜] β i) [DecidableEq ι]
    (i : ι) (v : α i) :
    LinearIsometryEquiv.piLpCongrRight p e ((WithLp.equiv p (∀ i, α i)).symm <| Pi.single i v) =
      (WithLp.equiv p (∀ i, β i)).symm (Pi.single i (e _ v)) :=
  funext <| Pi.apply_single (e ·) (fun _ => map_zero _) _ _


variable (𝕜) in
/-- `LinearEquiv.piCurry` for `PiLp`, as an isometry. -/
def _root_.LinearIsometryEquiv.piLpCurry :
    PiLp p (fun i : Sigma _ => α i.1 i.2) ≃ₗᵢ[𝕜] PiLp p (fun i => PiLp p (α i)) where
  toLinearEquiv :=
    WithLp.linearEquiv _ _ _
      ≪≫ₗ LinearEquiv.piCurry 𝕜 α
      ≪≫ₗ (LinearEquiv.piCongrRight fun _ => (WithLp.linearEquiv _ _ _).symm)
      ≪≫ₗ (WithLp.linearEquiv _ _ _).symm
  norm_map' := (WithLp.equiv p _).symm.surjective.forall.2 fun x => by
    /-
      p✝ : ENNReal
      𝕜 : Type u_1
      ι✝ : Type u_2
      α✝ : ι✝ → Type u_3
      β : ι✝ → Type u_4
      hp : Fact (LE.le 1 p✝)
      inst✝¹⁰ : Fintype ι✝
      inst✝⁹ : Semiring 𝕜
      inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
      inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
      inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
      inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
      c : 𝕜
      ι : Type u_5
      κ : ι → Type u_6
      p : ENNReal
      inst✝⁴ : Fact (LE.le 1 p)
      inst✝³ : Fintype ι
      inst✝² : (i : ι) → Fintype (κ i)
      α : (i : ι) → κ i → Type u_7
      inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
      inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
      x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
      ⊢ Eq (Norm.norm (((((WithLp.linearEquiv p 𝕜 ((i : Sigma κ) → (fun i => α i.fst …
    -/
    simp_rw [← coe_nnnorm, NNReal.coe_inj]
    /-
      p✝ : ENNReal
      𝕜 : Type u_1
      ι✝ : Type u_2
      α✝ : ι✝ → Type u_3
      β : ι✝ → Type u_4
      hp : Fact (LE.le 1 p✝)
      inst✝¹⁰ : Fintype ι✝
      inst✝⁹ : Semiring 𝕜
      inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
      inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
      inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
      inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
      c : 𝕜
      ι : Type u_5
      κ : ι → Type u_6
      p : ENNReal
      inst✝⁴ : Fact (LE.le 1 p)
      inst✝³ : Fintype ι
      inst✝² : (i : ι) → Fintype (κ i)
      α : (i : ι) → κ i → Type u_7
      inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
      inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
      x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
      ⊢ Eq (NNNorm.nnnorm (((((WithLp.linearEquiv p 𝕜 ((i : Sigma κ) → α i.fst i.snd …
    -/
    obtain rfl | hp := eq_or_ne p ⊤
      /-
        case inl
        p : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        inst✝⁴ : Fintype ι
        inst✝³ : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝² : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝¹ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        inst✝ : Fact (LE.le 1 Top.top)
        ⊢ Eq (NNNorm.nnnorm (((((WithLp.linearEquiv Top.top 𝕜 ((i : Sigma κ) → α i.fst …
      -/
    · simp_rw [← PiLp.nnnorm_equiv, Pi.nnnorm_def, ← PiLp.nnnorm_equiv, Pi.nnnorm_def]
      /-
        case inl
        p : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        inst✝⁴ : Fintype ι
        inst✝³ : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝² : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝¹ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        inst✝ : Fact (LE.le 1 Top.top)
        ⊢ Eq (Finset.univ.sup fun b => Finset.univ.sup fun b_1 => NNNorm.nnnorm ((With …
      -/
      dsimp [Sigma.curry]
      /-
        case inl
        p : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp : Fact (LE.le 1 p)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        inst✝⁴ : Fintype ι
        inst✝³ : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝² : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝¹ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        inst✝ : Fact (LE.le 1 Top.top)
        ⊢ Eq (Finset.univ.sup fun b => Finset.univ.sup fun b_1 => NNNorm.nnnorm (x ⟨b, …
      -/
      rw [← Finset.univ_sigma_univ, Finset.sup_sigma]
      /-
        🎉 no goals
      -/
      /-
        case inr
        p✝ : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp✝ : Fact (LE.le 1 p✝)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        p : ENNReal
        inst✝⁴ : Fact (LE.le 1 p)
        inst✝³ : Fintype ι
        inst✝² : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        hp : Ne p Top.top
        ⊢ Eq (NNNorm.nnnorm (((((WithLp.linearEquiv p 𝕜 ((i : Sigma κ) → α i.fst i.snd …
      -/
    · have : 0 < p.toReal := (toReal_pos_iff_ne_top _).mpr hp
      /-
        case inr
        p✝ : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp✝ : Fact (LE.le 1 p✝)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        p : ENNReal
        inst✝⁴ : Fact (LE.le 1 p)
        inst✝³ : Fintype ι
        inst✝² : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        hp : Ne p Top.top
        this : LT.lt 0 p.toReal
        ⊢ Eq (NNNorm.nnnorm (((((WithLp.linearEquiv p 𝕜 ((i : Sigma κ) → α i.fst i.snd …
      -/
      simp_rw [PiLp.nnnorm_eq_sum hp, WithLp.equiv_symm_pi_apply]
      /-
        case inr
        p✝ : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp✝ : Fact (LE.le 1 p✝)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        p : ENNReal
        inst✝⁴ : Fact (LE.le 1 p)
        inst✝³ : Fintype ι
        inst✝² : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        hp : Ne p Top.top
        this : LT.lt 0 p.toReal
        ⊢ Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow (HPow.hPow (Finset.univ. …
      -/
      dsimp [Sigma.curry]
      /-
        case inr
        p✝ : ENNReal
        𝕜 : Type u_1
        ι✝ : Type u_2
        α✝ : ι✝ → Type u_3
        β : ι✝ → Type u_4
        hp✝ : Fact (LE.le 1 p✝)
        inst✝¹⁰ : Fintype ι✝
        inst✝⁹ : Semiring 𝕜
        inst✝⁸ : (i : ι✝) → SeminormedAddCommGroup (α✝ i)
        inst✝⁷ : (i : ι✝) → SeminormedAddCommGroup (β i)
        inst✝⁶ : (i : ι✝) → Module 𝕜 (α✝ i)
        inst✝⁵ : (i : ι✝) → Module 𝕜 (β i)
        c : 𝕜
        ι : Type u_5
        κ : ι → Type u_6
        p : ENNReal
        inst✝⁴ : Fact (LE.le 1 p)
        inst✝³ : Fintype ι
        inst✝² : (i : ι) → Fintype (κ i)
        α : (i : ι) → κ i → Type u_7
        inst✝¹ : (i : ι) → (k : κ i) → SeminormedAddCommGroup (α i k)
        inst✝ : (i : ι) → (k : κ i) → Module 𝕜 (α i k)
        x : (i : Sigma κ) → (fun i => α i.fst i.snd) i
        hp : Ne p Top.top
        this : LT.lt 0 p.toReal
        ⊢ Eq (HPow.hPow (Finset.univ.sum fun x_1 => HPow.hPow (HPow.hPow (Finset.univ. …
      -/
      simp_rw [one_div, NNReal.rpow_inv_rpow this.ne', ← Finset.univ_sigma_univ, Finset.sum_sigma]
      /-
        🎉 no goals
      -/


@[simp] theorem _root_.LinearIsometryEquiv.piLpCurry_apply
    (f : PiLp p (fun i : Sigma κ => α i.1 i.2)) :
    _root_.LinearIsometryEquiv.piLpCurry 𝕜 p α f =
      (WithLp.equiv _ _).symm (fun i => (WithLp.equiv _ _).symm <|
        Sigma.curry (WithLp.equiv _ _ f) i) :=
  rfl


@[simp] theorem _root_.LinearIsometryEquiv.piLpCurry_symm_apply
    (f : PiLp p (fun i => PiLp p (α i))) :
    (_root_.LinearIsometryEquiv.piLpCurry 𝕜 p α).symm f =
      (WithLp.equiv _ _).symm (Sigma.uncurry fun i j => f i j) :=
  rfl


@[simp]
theorem nnnorm_equiv_symm_single (i : ι) (b : β i) :
    ‖(WithLp.equiv p (∀ i, β i)).symm (Pi.single i b)‖₊ = ‖b‖₊ := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    hp : Fact (LE.le 1 p)
    inst✝² : Fintype ι
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (β i)
    inst✝ : DecidableEq ι
    i : ι
    b : β i
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p ((i : ι) → β i)).symm (Pi.single i b))) ( …
  -/
  haveI : Nonempty ι := ⟨i⟩
  induction p generalizing hp with
  | top =>
    simp_rw [nnnorm_eq_ciSup, WithLp.equiv_symm_pi_apply]
    refine
      ciSup_eq_of_forall_le_of_forall_lt_exists_gt (fun j => ?_) fun n hn => ⟨i, hn.trans_eq ?_⟩
    · obtain rfl | hij := Decidable.eq_or_ne i j
      · rw [Pi.single_eq_same]
      · rw [Pi.single_eq_of_ne' hij, nnnorm_zero]
        exact zero_le _
    · rw [Pi.single_eq_same]
  | coe p =>
    have hp0 : (p : ℝ) ≠ 0 :=
      mod_cast (zero_lt_one.trans_le <| Fact.out (p := 1 ≤ (p : ℝ≥0∞))).ne'
    rw [nnnorm_eq_sum ENNReal.coe_ne_top, ENNReal.coe_toReal, Fintype.sum_eq_single i,
      WithLp.equiv_symm_pi_apply, Pi.single_eq_same, ← NNReal.rpow_mul, one_div,
      mul_inv_cancel₀ hp0, NNReal.rpow_one]
    intro j hij
    rw [WithLp.equiv_symm_pi_apply, Pi.single_eq_of_ne hij, nnnorm_zero, NNReal.zero_rpow hp0]


@[simp]
theorem norm_equiv_symm_single (i : ι) (b : β i) :
    ‖(WithLp.equiv p (∀ i, β i)).symm (Pi.single i b)‖ = ‖b‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_equiv_symm_single p β i b


@[simp]
theorem nndist_equiv_symm_single_same (i : ι) (b₁ b₂ : β i) :
    nndist
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₁))
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₂)) =
      nndist b₁ b₂ := by
  rw [nndist_eq_nnnorm, nndist_eq_nnnorm, ← WithLp.equiv_symm_sub, ← Pi.single_sub,
    nnnorm_equiv_symm_single]


@[simp]
theorem dist_equiv_symm_single_same (i : ι) (b₁ b₂ : β i) :
    dist
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₁))
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₂)) =
      dist b₁ b₂ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nndist_equiv_symm_single_same p β i b₁ b₂


@[simp]
theorem edist_equiv_symm_single_same (i : ι) (b₁ b₂ : β i) :
    edist
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₁))
        ((WithLp.equiv p (∀ i, β i)).symm (Pi.single i b₂)) =
      edist b₁ b₂ := by
  /-
    p : ENNReal
    ι : Type u_2
    β : ι → Type u_4
    hp : Fact (LE.le 1 p)
    inst✝² : Fintype ι
    inst✝¹ : (i : ι) → SeminormedAddCommGroup (β i)
    inst✝ : DecidableEq ι
    i : ι
    b₁ b₂ : β i
    ⊢ Eq (EDist.edist ((WithLp.equiv p ((i : ι) → β i)).symm (Pi.single i b₁)) ((W …
  -/
  simp only [edist_nndist, nndist_equiv_symm_single_same p β i b₁ b₂]
  /-
    🎉 no goals
  -/


/-- When `p = ∞`, this lemma does not hold without the additional assumption `Nonempty ι` because
the left-hand side simplifies to `0`, while the right-hand side simplifies to `‖b‖₊`. See
`PiLp.nnnorm_equiv_symm_const'` for a version which exchanges the hypothesis `p ≠ ∞` for
`Nonempty ι`. -/
theorem nnnorm_equiv_symm_const {β} [SeminormedAddCommGroup β] (hp : p ≠ ∞) (b : β) :
    ‖(WithLp.equiv p (ι → β)).symm (Function.const _ b)‖₊ =
      (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖b‖₊ := by
  /-
    p : ENNReal
    ι : Type u_2
    hp✝ : Fact (LE.le 1 p)
    inst✝¹ : Fintype ι
    β : Type u_5
    inst✝ : SeminormedAddCommGroup β
    hp : Ne p Top.top
    b : β
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p (ι → β)).symm (Function.const ι b))) (HMu …
  -/
  rcases p.dichotomy with (h | h)
    /-
      case inl
      p : ENNReal
      ι : Type u_2
      hp✝ : Fact (LE.le 1 p)
      inst✝¹ : Fintype ι
      β : Type u_5
      inst✝ : SeminormedAddCommGroup β
      hp : Ne p Top.top
      b : β
      h : Eq p Top.top
      ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p (ι → β)).symm (Function.const ι b))) (HMu …
    -/
  · exact False.elim (hp h)
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      hp✝ : Fact (LE.le 1 p)
      inst✝¹ : Fintype ι
      β : Type u_5
      inst✝ : SeminormedAddCommGroup β
      hp : Ne p Top.top
      b : β
      h : LE.le 1 p.toReal
      ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p (ι → β)).symm (Function.const ι b))) (HMu …
    -/
  · have ne_zero : p.toReal ≠ 0 := (zero_lt_one.trans_le h).ne'
    simp_rw [nnnorm_eq_sum hp, WithLp.equiv_symm_pi_apply, Function.const_apply, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul, NNReal.mul_rpow, ← NNReal.rpow_mul,
      mul_one_div_cancel ne_zero, NNReal.rpow_one, ENNReal.toReal_div, ENNReal.one_toReal]


/-- When `IsEmpty ι`, this lemma does not hold without the additional assumption `p ≠ ∞` because
the left-hand side simplifies to `0`, while the right-hand side simplifies to `‖b‖₊`. See
`PiLp.nnnorm_equiv_symm_const` for a version which exchanges the hypothesis `Nonempty ι`.
for `p ≠ ∞`. -/
theorem nnnorm_equiv_symm_const' {β} [SeminormedAddCommGroup β] [Nonempty ι] (b : β) :
    ‖(WithLp.equiv p (ι → β)).symm (Function.const _ b)‖₊ =
      (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖b‖₊ := by
  /-
    p : ENNReal
    ι : Type u_2
    hp : Fact (LE.le 1 p)
    inst✝² : Fintype ι
    β : Type u_5
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : Nonempty ι
    b : β
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p (ι → β)).symm (Function.const ι b))) (HMu …
  -/
  rcases em <| p = ∞ with (rfl | hp)
  · simp only [WithLp.equiv_symm_pi_apply, ENNReal.div_top, ENNReal.zero_toReal, NNReal.rpow_zero,
      one_mul, nnnorm_eq_ciSup, Function.const_apply, ciSup_const]
    /-
      case inr
      p : ENNReal
      ι : Type u_2
      hp✝ : Fact (LE.le 1 p)
      inst✝² : Fintype ι
      β : Type u_5
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : Nonempty ι
      b : β
      hp : Not (Eq p Top.top)
      ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv p (ι → β)).symm (Function.const ι b))) (HMu …
    -/
  · exact nnnorm_equiv_symm_const hp b
    /-
      🎉 no goals
    -/


/-- When `p = ∞`, this lemma does not hold without the additional assumption `Nonempty ι` because
the left-hand side simplifies to `0`, while the right-hand side simplifies to `‖b‖₊`. See
`PiLp.norm_equiv_symm_const'` for a version which exchanges the hypothesis `p ≠ ∞` for
`Nonempty ι`. -/
theorem norm_equiv_symm_const {β} [SeminormedAddCommGroup β] (hp : p ≠ ∞) (b : β) :
    ‖(WithLp.equiv p (ι → β)).symm (Function.const _ b)‖ =
      (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖b‖ :=
                                                                          /-
                                                                            p : ENNReal
                                                                            ι : Type u_2
                                                                            hp✝ : Fact (LE.le 1 p)
                                                                            inst✝¹ : Fintype ι
                                                                            β : Type u_5
                                                                            inst✝ : SeminormedAddCommGroup β
                                                                            hp : Ne p Top.top
                                                                            b : β
                                                                            ⊢ Eq (↑(HMul.hMul (HPow.hPow (↑(Fintype.card ι)) (HDiv.hDiv 1 p).toReal) (NNNo …
                                                                          -/
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_equiv_symm_const hp b).trans <| by simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


/-- When `IsEmpty ι`, this lemma does not hold without the additional assumption `p ≠ ∞` because
the left-hand side simplifies to `0`, while the right-hand side simplifies to `‖b‖₊`. See
`PiLp.norm_equiv_symm_const` for a version which exchanges the hypothesis `Nonempty ι`.
for `p ≠ ∞`. -/
theorem norm_equiv_symm_const' {β} [SeminormedAddCommGroup β] [Nonempty ι] (b : β) :
    ‖(WithLp.equiv p (ι → β)).symm (Function.const _ b)‖ =
      (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖b‖ :=
                                                                        /-
                                                                          p : ENNReal
                                                                          ι : Type u_2
                                                                          hp : Fact (LE.le 1 p)
                                                                          inst✝² : Fintype ι
                                                                          β : Type u_5
                                                                          inst✝¹ : SeminormedAddCommGroup β
                                                                          inst✝ : Nonempty ι
                                                                          b : β
                                                                          ⊢ Eq (↑(HMul.hMul (HPow.hPow (↑(Fintype.card ι)) (HDiv.hDiv 1 p).toReal) (NNNo …
                                                                        -/
  (congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_equiv_symm_const' b).trans <| by simp
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem nnnorm_equiv_symm_one {β} [SeminormedAddCommGroup β] (hp : p ≠ ∞) [One β] :
    ‖(WithLp.equiv p (ι → β)).symm 1‖₊ =
      (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖(1 : β)‖₊ :=
  (nnnorm_equiv_symm_const hp (1 : β)).trans rfl


theorem norm_equiv_symm_one {β} [SeminormedAddCommGroup β] (hp : p ≠ ∞) [One β] :
    ‖(WithLp.equiv p (ι → β)).symm 1‖ = (Fintype.card ι : ℝ≥0) ^ (1 / p).toReal * ‖(1 : β)‖ :=
  (norm_equiv_symm_const hp (1 : β)).trans rfl


/-- `WithLp.equiv` as a continuous linear equivalence. -/
@[simps! (config := .asFn) apply symm_apply]
protected def continuousLinearEquiv : PiLp p β ≃L[𝕜] ∀ i, β i where
  toLinearEquiv := WithLp.linearEquiv _ _ _
  continuous_toFun := continuous_equiv _ _
  continuous_invFun := continuous_equiv_symm _ _


variable {𝕜} in
/-- The projection on the `i`-th coordinate of `PiLp p β`, as a continuous linear map. -/
@[simps!]
def proj (i : ι) : PiLp p β →L[𝕜] β i where
  __ := projₗ p β i
  cont := continuous_apply i


/-- A version of `Pi.basisFun` for `PiLp`. -/
def basisFun : Basis ι 𝕜 (PiLp p fun _ : ι => 𝕜) :=
  Basis.ofEquivFun (WithLp.linearEquiv p 𝕜 (ι → 𝕜))


@[simp]
theorem basisFun_apply [DecidableEq ι] (i) :
    basisFun p 𝕜 ι i = (WithLp.equiv p _).symm (Pi.single i 1) := by
  /-
    p : ENNReal
    𝕜 : Type u_1
    ι : Type u_2
    inst✝² : Finite ι
    inst✝¹ : Ring 𝕜
    inst✝ : DecidableEq ι
    i : ι
    ⊢ Eq ((PiLp.basisFun p 𝕜 ι) i) ((WithLp.equiv p (ι → 𝕜)).symm (Pi.single i 1))
  -/
  simp_rw [basisFun, Basis.coe_ofEquivFun, WithLp.linearEquiv_symm_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem basisFun_repr (x : PiLp p fun _ : ι => 𝕜) (i : ι) : (basisFun p 𝕜 ι).repr x i = x i :=
  rfl


@[simp]
theorem basisFun_equivFun : (basisFun p 𝕜 ι).equivFun = WithLp.linearEquiv p 𝕜 (ι → 𝕜) :=
  Basis.equivFun_ofEquivFun _


theorem basisFun_eq_pi_basisFun :
    basisFun p 𝕜 ι = (Pi.basisFun 𝕜 ι).map (WithLp.linearEquiv p 𝕜 (ι → 𝕜)).symm :=
  rfl


@[simp]
theorem basisFun_map :
    (basisFun p 𝕜 ι).map (WithLp.linearEquiv p 𝕜 (ι → 𝕜)) = Pi.basisFun 𝕜 ι :=
  rfl


nonrec theorem basis_toMatrix_basisFun_mul [Fintype ι]
    {𝕜} [SeminormedCommRing 𝕜] (b : Basis ι 𝕜 (PiLp p fun _ : ι => 𝕜))
    (A : Matrix ι ι 𝕜) :
    b.toMatrix (PiLp.basisFun _ _ _) * A =
      Matrix.of fun i j => b.repr ((WithLp.equiv _ _).symm (Aᵀ j)) i := by
  /-
    p : ENNReal
    ι : Type u_2
    inst✝¹ : Fintype ι
    𝕜 : Type u_5
    inst✝ : SeminormedCommRing 𝕜
    b : Basis ι 𝕜 (PiLp p fun x => 𝕜)
    A : Matrix ι ι 𝕜
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑(PiLp.basisFun p 𝕜 ι)) A) (Matrix.of fun i j => ( …
  -/
  have := basis_toMatrix_basisFun_mul (b.map (WithLp.linearEquiv _ 𝕜 _)) A
  simp_rw [← PiLp.basisFun_map p, Basis.map_repr, LinearEquiv.trans_apply,
    WithLp.linearEquiv_symm_apply, Basis.toMatrix_map, Function.comp_def, Basis.map_apply,
    LinearEquiv.symm_apply_apply] at this
  /-
    p : ENNReal
    ι : Type u_2
    inst✝¹ : Fintype ι
    𝕜 : Type u_5
    inst✝ : SeminormedCommRing 𝕜
    b : Basis ι 𝕜 (PiLp p fun x => 𝕜)
    A : Matrix ι ι 𝕜
    this : Eq (HMul.hMul (b.toMatrix fun x => (PiLp.basisFun p 𝕜 ι) x) A) (Matrix. …
    ⊢ Eq (HMul.hMul (b.toMatrix ⇑(PiLp.basisFun p 𝕜 ι)) A) (Matrix.of fun i j => ( …
  -/
  exact this
  /-
    🎉 no goals
  -/


