/-- Let `V` be a real topological vector space. A subset of `V` is a convex body if and only if
it is convex, compact, and nonempty.
-/
structure ConvexBody (V : Type*) [TopologicalSpace V] [AddCommMonoid V] [SMul ℝ V] where
  /-- The **carrier set** underlying a convex body: the set of points contained in it -/
  carrier : Set V
  /-- A convex body has convex carrier set -/
  convex' : Convex ℝ carrier
  /-- A convex body has compact carrier set -/
  isCompact' : IsCompact carrier
  /-- A convex body has non-empty carrier set -/
  nonempty' : carrier.Nonempty


instance : SetLike (ConvexBody V) V where
  coe := ConvexBody.carrier
  coe_injective' K L h := by
    /-
      V : Type u_1
      inst✝² : TopologicalSpace V
      inst✝¹ : AddCommGroup V
      inst✝ : Module Real V
      K L : ConvexBody V
      h : Eq K.carrier L.carrier
      ⊢ Eq K L
    -/
    cases K
    /-
      case mk
      V : Type u_1
      inst✝² : TopologicalSpace V
      inst✝¹ : AddCommGroup V
      inst✝ : Module Real V
      L : ConvexBody V
      carrier✝ : Set V
      convex'✝ : Convex Real carrier✝
      isCompact'✝ : IsCompact carrier✝
      nonempty'✝ : carrier✝.Nonempty
      h : Eq { carrier := carrier✝, convex' := convex'✝, isCompact' := isCompact'✝,  …
      ⊢ Eq { carrier := carrier✝, convex' := convex'✝, isCompact' := isCompact'✝, no …
    -/
    cases L
    /-
      case mk.mk
      V : Type u_1
      inst✝² : TopologicalSpace V
      inst✝¹ : AddCommGroup V
      inst✝ : Module Real V
      carrier✝¹ : Set V
      convex'✝¹ : Convex Real carrier✝¹
      isCompact'✝¹ : IsCompact carrier✝¹
      nonempty'✝¹ : carrier✝¹.Nonempty
      carrier✝ : Set V
      convex'✝ : Convex Real carrier✝
      isCompact'✝ : IsCompact carrier✝
      nonempty'✝ : carrier✝.Nonempty
      h : Eq { carrier := carrier✝¹, convex' := convex'✝¹, isCompact' := isCompact'✝ …
      ⊢ Eq { carrier := carrier✝¹, convex' := convex'✝¹, isCompact' := isCompact'✝¹, …
    -/
    congr
    /-
      🎉 no goals
    -/


protected theorem convex (K : ConvexBody V) : Convex ℝ (K : Set V) :=
  K.convex'


protected theorem isCompact (K : ConvexBody V) : IsCompact (K : Set V) :=
  K.isCompact'


protected theorem isClosed [T2Space V] (K : ConvexBody V) : IsClosed (K : Set V) :=
  K.isCompact.isClosed


protected theorem nonempty (K : ConvexBody V) : (K : Set V).Nonempty :=
  K.nonempty'


@[ext]
protected theorem ext {K L : ConvexBody V} (h : (K : Set V) = L) : K = L :=
  SetLike.ext' h


@[simp]
theorem coe_mk (s : Set V) (h₁ h₂ h₃) : (mk s h₁ h₂ h₃ : Set V) = s :=
  rfl


/-- A convex body that is symmetric contains `0`. -/
theorem zero_mem_of_symmetric (K : ConvexBody V) (h_symm : ∀ x ∈ K, - x ∈ K) : 0 ∈ K := by
  /-
    V : Type u_1
    inst✝² : TopologicalSpace V
    inst✝¹ : AddCommGroup V
    inst✝ : Module Real V
    K : ConvexBody V
    h_symm : ∀ (x : V), Membership.mem K x → Membership.mem K (Neg.neg x)
    ⊢ Membership.mem K 0
  -/
  obtain ⟨x, hx⟩ := K.nonempty
  /-
    case intro
    V : Type u_1
    inst✝² : TopologicalSpace V
    inst✝¹ : AddCommGroup V
    inst✝ : Module Real V
    K : ConvexBody V
    h_symm : ∀ (x : V), Membership.mem K x → Membership.mem K (Neg.neg x)
    x : V
    hx : Membership.mem (↑K) x
    ⊢ Membership.mem K 0
  -/
  rw [show 0 = (1/2 : ℝ) • x + (1/2 : ℝ) • (- x) by field_simp]
  /-
    case intro
    V : Type u_1
    inst✝² : TopologicalSpace V
    inst✝¹ : AddCommGroup V
    inst✝ : Module Real V
    K : ConvexBody V
    h_symm : ∀ (x : V), Membership.mem K x → Membership.mem K (Neg.neg x)
    x : V
    hx : Membership.mem (↑K) x
    ⊢ Membership.mem K (HAdd.hAdd (HSMul.hSMul (1 / 2) x) (HSMul.hSMul (1 / 2) (Ne …
  -/
  apply convex_iff_forall_pos.mp K.convex hx (h_symm x hx)
  /-
    case intro.a
    V : Type u_1
    inst✝² : TopologicalSpace V
    inst✝¹ : AddCommGroup V
    inst✝ : Module Real V
    K : ConvexBody V
    h_symm : ∀ (x : V), Membership.mem K x → Membership.mem K (Neg.neg x)
    x : V
    hx : Membership.mem (↑K) x
    ⊢ LT.lt 0 (1 / 2)
  -/
  all_goals linarith
  /-
    🎉 no goals
  -/


instance : Zero (ConvexBody V) where
  zero := ⟨0, convex_singleton 0, isCompact_singleton, Set.singleton_nonempty 0⟩


@[simp] -- Porting note: add norm_cast; we leave it out for now to reproduce mathlib3 behavior.
theorem coe_zero : (↑(0 : ConvexBody V) : Set V) = 0 :=
  rfl


instance : Inhabited (ConvexBody V) :=
  ⟨0⟩


instance : Add (ConvexBody V) where
  add K L :=
    ⟨K + L, K.convex.add L.convex, K.isCompact.add L.isCompact,
      K.nonempty.add L.nonempty⟩


instance : SMul ℕ (ConvexBody V) where
  smul := nsmulRec

-- Porting note: add @[simp, norm_cast]; we leave it out for now to reproduce mathlib3 behavior.

theorem coe_nsmul : ∀ (n : ℕ) (K : ConvexBody V), ↑(n • K) = n • (K : Set V)
  | 0, _ => rfl
  | (n + 1), K => congr_arg₂ (Set.image2 (· + ·)) (coe_nsmul n K) rfl


instance : AddMonoid (ConvexBody V) :=
  SetLike.coe_injective.addMonoid _ rfl (fun _ _ ↦ rfl) fun _ _ ↦ coe_nsmul _ _


@[simp] -- Porting note: add norm_cast; we leave it out for now to reproduce mathlib3 behavior.
theorem coe_add (K L : ConvexBody V) : (↑(K + L) : Set V) = (K : Set V) + L :=
  rfl


instance : AddCommMonoid (ConvexBody V) :=
  SetLike.coe_injective.addCommMonoid _ rfl (fun _ _ ↦ rfl) fun _ _ ↦ coe_nsmul _ _


instance : SMul ℝ (ConvexBody V) where
  smul c K := ⟨c • (K : Set V), K.convex.smul _, K.isCompact.smul _, K.nonempty.smul_set⟩


@[simp] -- Porting note: add norm_cast; we leave it out for now to reproduce mathlib3 behavior.
theorem coe_smul (c : ℝ) (K : ConvexBody V) : (↑(c • K) : Set V) = c • (K : Set V) :=
  rfl


instance : DistribMulAction ℝ (ConvexBody V) :=
  SetLike.coe_injective.distribMulAction ⟨⟨_, coe_zero⟩, coe_add⟩ coe_smul


@[simp] -- Porting note: add norm_cast; we leave it out for now to reproduce mathlib3 behavior.
theorem coe_smul' (c : ℝ≥0) (K : ConvexBody V) : (↑(c • K) : Set V) = c • (K : Set V) :=
  rfl


/-- The convex bodies in a fixed space $V$ form a module over the nonnegative reals.
-/
instance : Module ℝ≥0 (ConvexBody V) where
  add_smul c d K := SetLike.ext' <| Convex.add_smul K.convex c.coe_nonneg d.coe_nonneg
  zero_smul K := SetLike.ext' <| Set.zero_smul_set K.nonempty


theorem smul_le_of_le (K : ConvexBody V) (h_zero : 0 ∈ K) {a b : ℝ≥0} (h : a ≤ b) :
    a • K ≤ b • K := by
  /-
    V : Type u_1
    inst✝⁴ : TopologicalSpace V
    inst✝³ : AddCommGroup V
    inst✝² : Module Real V
    inst✝¹ : ContinuousSMul Real V
    inst✝ : ContinuousAdd V
    K : ConvexBody V
    h_zero : Membership.mem K 0
    a b : NNReal
    h : LE.le a b
    ⊢ LE.le (HSMul.hSMul a K) (HSMul.hSMul b K)
  -/
  rw [← SetLike.coe_subset_coe, coe_smul', coe_smul']
  /-
    V : Type u_1
    inst✝⁴ : TopologicalSpace V
    inst✝³ : AddCommGroup V
    inst✝² : Module Real V
    inst✝¹ : ContinuousSMul Real V
    inst✝ : ContinuousAdd V
    K : ConvexBody V
    h_zero : Membership.mem K 0
    a b : NNReal
    h : LE.le a b
    ⊢ HasSubset.Subset (HSMul.hSMul a ↑K) (HSMul.hSMul b ↑K)
  -/
  obtain rfl | ha := eq_zero_or_pos a
    /-
      case inl
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      b : NNReal
      h : LE.le 0 b
      ⊢ HasSubset.Subset (HSMul.hSMul 0 ↑K) (HSMul.hSMul b ↑K)
    -/
  · rw [Set.zero_smul_set K.nonempty, Set.zero_subset]
    /-
      case inl
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      b : NNReal
      h : LE.le 0 b
      ⊢ Membership.mem (HSMul.hSMul b ↑K) 0
    -/
    exact Set.mem_smul_set.mpr ⟨0, h_zero, smul_zero _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      a b : NNReal
      h : LE.le a b
      ha : LT.lt 0 a
      ⊢ HasSubset.Subset (HSMul.hSMul a ↑K) (HSMul.hSMul b ↑K)
    -/
  · intro x hx
    /-
      case inr
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      a b : NNReal
      h : LE.le a b
      ha : LT.lt 0 a
      x : V
      hx : Membership.mem (HSMul.hSMul a ↑K) x
      ⊢ Membership.mem (HSMul.hSMul b ↑K) x
    -/
    obtain ⟨y, hy, rfl⟩ := Set.mem_smul_set.mp hx
    /-
      case inr.intro.intro
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      a b : NNReal
      h : LE.le a b
      ha : LT.lt 0 a
      y : V
      hy : Membership.mem (↑K) y
      hx : Membership.mem (HSMul.hSMul a ↑K) (HSMul.hSMul a y)
      ⊢ Membership.mem (HSMul.hSMul b ↑K) (HSMul.hSMul a y)
    -/
    rw [← Set.mem_inv_smul_set_iff₀ ha.ne', smul_smul]
    /-
      case inr.intro.intro
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      a b : NNReal
      h : LE.le a b
      ha : LT.lt 0 a
      y : V
      hy : Membership.mem (↑K) y
      hx : Membership.mem (HSMul.hSMul a ↑K) (HSMul.hSMul a y)
      ⊢ Membership.mem (HSMul.hSMul (HMul.hMul (Inv.inv a) b) ↑K) y
    -/
    refine Convex.mem_smul_of_zero_mem K.convex h_zero hy (?_ : 1 ≤ a⁻¹ * b)
    /-
      case inr.intro.intro
      V : Type u_1
      inst✝⁴ : TopologicalSpace V
      inst✝³ : AddCommGroup V
      inst✝² : Module Real V
      inst✝¹ : ContinuousSMul Real V
      inst✝ : ContinuousAdd V
      K : ConvexBody V
      h_zero : Membership.mem K 0
      a b : NNReal
      h : LE.le a b
      ha : LT.lt 0 a
      y : V
      hy : Membership.mem (↑K) y
      hx : Membership.mem (HSMul.hSMul a ↑K) (HSMul.hSMul a y)
      ⊢ LE.le 1 (HMul.hMul (Inv.inv a) b)
    -/
    rwa [le_inv_mul_iff₀ ha, mul_one]
    /-
      🎉 no goals
    -/


protected theorem isBounded : Bornology.IsBounded (K : Set V) :=
  K.isCompact.isBounded


theorem hausdorffEdist_ne_top {K L : ConvexBody V} : EMetric.hausdorffEdist (K : Set V) L ≠ ⊤ := by
  apply_rules [Metric.hausdorffEdist_ne_top_of_nonempty_of_bounded, ConvexBody.nonempty,
    ConvexBody.isBounded]


/-- Convex bodies in a fixed seminormed space $V$ form a pseudo-metric space under the Hausdorff
metric. -/
noncomputable instance : PseudoMetricSpace (ConvexBody V) where
  dist K L := Metric.hausdorffDist (K : Set V) L
  dist_self _ := Metric.hausdorffDist_self_zero
  dist_comm _ _ := Metric.hausdorffDist_comm
  dist_triangle _ _ _ := Metric.hausdorffDist_triangle hausdorffEdist_ne_top


@[simp, norm_cast]
theorem hausdorffDist_coe : Metric.hausdorffDist (K : Set V) L = dist K L :=
  rfl


@[simp, norm_cast]
theorem hausdorffEdist_coe : EMetric.hausdorffEdist (K : Set V) L = edist K L := by
  /-
    V : Type u_1
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : NormedSpace Real V
    K L : ConvexBody V
    ⊢ Eq (EMetric.hausdorffEdist ↑K ↑L) (EDist.edist K L)
  -/
  rw [edist_dist]
  /-
    V : Type u_1
    inst✝¹ : SeminormedAddCommGroup V
    inst✝ : NormedSpace Real V
    K L : ConvexBody V
    ⊢ Eq (EMetric.hausdorffEdist ↑K ↑L) (ENNReal.ofReal (Dist.dist K L))
  -/
  exact (ENNReal.ofReal_toReal hausdorffEdist_ne_top).symm
  /-
    🎉 no goals
  -/


/-- Let `K` be a convex body that contains `0` and let `u n` be a sequence of nonnegative real
numbers that tends to `0`. Then the intersection of the dilated bodies `(1 + u n) • K` is equal
to `K`. -/
theorem iInter_smul_eq_self [T2Space V] {u : ℕ → ℝ≥0} (K : ConvexBody V) (h_zero : 0 ∈ K)
    (hu : Tendsto u atTop (𝓝 0)) :
    ⋂ n : ℕ, (1 + (u n : ℝ)) • (K : Set V) = K := by
  /-
    V : Type u_1
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : T2Space V
    u : Nat → NNReal
    K : ConvexBody V
    h_zero : Membership.mem K 0
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    ⊢ Eq (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ↑K
  -/
  ext x
  /-
    case h
    V : Type u_1
    inst✝² : SeminormedAddCommGroup V
    inst✝¹ : NormedSpace Real V
    inst✝ : T2Space V
    u : Nat → NNReal
    K : ConvexBody V
    h_zero : Membership.mem K 0
    hu : Filter.Tendsto u Filter.atTop (nhds 0)
    x : V
    ⊢ Iff (Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case h.refine_1
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      ⊢ Membership.mem (↑K) x
    -/
  · obtain ⟨C, hC_pos, hC_bdd⟩ := K.isBounded.exists_pos_norm_le
    /-
      case h.refine_1.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ⊢ Membership.mem (↑K) x
    -/
    rw [← K.isClosed.closure_eq, SeminormedAddCommGroup.mem_closure_iff]
    /-
      case h.refine_1.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem (↑K) b) (LT.lt …
    -/
    rw [← NNReal.tendsto_coe, NormedAddCommGroup.tendsto_atTop] at hu
    /-
      case h.refine_1.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ⊢ ∀ (ε : Real), LT.lt 0 ε → Exists fun b => And (Membership.mem (↑K) b) (LT.lt …
    -/
    intro ε hε
    /-
      case h.refine_1.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Exists fun b => And (Membership.mem (↑K) b) (LT.lt (Norm.norm (HSub.hSub x b …
    -/
    obtain ⟨n, hn⟩ := hu (ε / C) (div_pos hε hC_pos)
    /-
      case h.refine_1.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      x : V
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      hn : ∀ (n_1 : Nat), LE.le n n_1 → LT.lt (Norm.norm (HSub.hSub ↑(u n_1) ↑0)) (H …
      ⊢ Exists fun b => And (Membership.mem (↑K) b) (LT.lt (Norm.norm (HSub.hSub x b …
    -/
    obtain ⟨y, hyK, rfl⟩ := Set.mem_smul_set.mp (Set.mem_iInter.mp h n)
    /-
      case h.refine_1.intro.intro.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      hn : ∀ (n_1 : Nat), LE.le n n_1 → LT.lt (Norm.norm (HSub.hSub ↑(u n_1) ↑0)) (H …
      y : V
      hyK : Membership.mem (↑K) y
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ( …
      ⊢ Exists fun b => And (Membership.mem (↑K) b) (LT.lt (Norm.norm (HSub.hSub (HS …
    -/
    refine ⟨y, hyK, ?_⟩
    rw [show (1 + u n : ℝ) • y - y = (u n : ℝ) • y by rw [add_smul, one_smul, add_sub_cancel_left],
      norm_smul, Real.norm_eq_abs]
    /-
      case h.refine_1.intro.intro.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      hn : ∀ (n_1 : Nat), LE.le n n_1 → LT.lt (Norm.norm (HSub.hSub ↑(u n_1) ↑0)) (H …
      y : V
      hyK : Membership.mem (↑K) y
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ( …
      ⊢ LT.lt (HMul.hMul (abs ↑(u n)) (Norm.norm y)) ε
    -/
    specialize hn n le_rfl
    /-
      case h.refine_1.intro.intro.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      y : V
      hyK : Membership.mem (↑K) y
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ( …
      hn : LT.lt (Norm.norm (HSub.hSub ↑(u n) ↑0)) (HDiv.hDiv ε C)
      ⊢ LT.lt (HMul.hMul (abs ↑(u n)) (Norm.norm y)) ε
    -/
    rw [lt_div_iff₀' hC_pos, mul_comm, NNReal.coe_zero, sub_zero, Real.norm_eq_abs] at hn
    /-
      case h.refine_1.intro.intro.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      y : V
      hyK : Membership.mem (↑K) y
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ( …
      hn : LT.lt (HMul.hMul (abs ↑(u n)) C) ε
      ⊢ LT.lt (HMul.hMul (abs ↑(u n)) (Norm.norm y)) ε
    -/
    refine lt_of_le_of_lt ?_ hn
    /-
      case h.refine_1.intro.intro.intro.intro.intro
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : ∀ (ε : Real), LT.lt 0 ε → Exists fun N => ∀ (n : Nat), LE.le N n → LT.lt  …
      C : Real
      hC_pos : GT.gt C 0
      hC_bdd : ∀ (x : V), Membership.mem (↑K) x → LE.le (Norm.norm x) C
      ε : Real
      hε : LT.lt 0 ε
      n : Nat
      y : V
      hyK : Membership.mem (↑K) y
      h : Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) ( …
      hn : LT.lt (HMul.hMul (abs ↑(u n)) C) ε
      ⊢ LE.le (HMul.hMul (abs ↑(u n)) (Norm.norm y)) (HMul.hMul (abs ↑(u n)) C)
    -/
    exact mul_le_mul_of_nonneg_left (hC_bdd _ hyK) (abs_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      x : V
      h : Membership.mem (↑K) x
      ⊢ Membership.mem (Set.iInter fun n => HSMul.hSMul (HAdd.hAdd 1 ↑(u n)) ↑K) x
    -/
  · refine Set.mem_iInter.mpr (fun n => Convex.mem_smul_of_zero_mem K.convex h_zero h ?_)
    /-
      case h.refine_2
      V : Type u_1
      inst✝² : SeminormedAddCommGroup V
      inst✝¹ : NormedSpace Real V
      inst✝ : T2Space V
      u : Nat → NNReal
      K : ConvexBody V
      h_zero : Membership.mem K 0
      hu : Filter.Tendsto u Filter.atTop (nhds 0)
      x : V
      h : Membership.mem (↑K) x
      n : Nat
      ⊢ LE.le 1 (HAdd.hAdd 1 ↑(u n))
    -/
    exact le_add_of_nonneg_right (by positivity)
    /-
      🎉 no goals
    -/


/-- Convex bodies in a fixed normed space `V` form a metric space under the Hausdorff metric. -/
noncomputable instance : MetricSpace (ConvexBody V) where
  eq_of_dist_eq_zero {K L} hd := ConvexBody.ext <|
    (K.isClosed.hausdorffDist_zero_iff_eq L.isClosed hausdorffEdist_ne_top).1 hd


