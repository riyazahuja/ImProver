@[simp]
theorem zero_fst : (0 : WithLp p (α × β)).fst = 0 :=
  rfl


@[simp]
theorem zero_snd : (0 : WithLp p (α × β)).snd = 0 :=
  rfl


@[simp]
theorem add_fst : (x + y).fst = x.fst + y.fst :=
  rfl


@[simp]
theorem add_snd : (x + y).snd = x.snd + y.snd :=
  rfl


@[simp]
theorem sub_fst : (x - y).fst = x.fst - y.fst :=
  rfl


@[simp]
theorem sub_snd : (x - y).snd = x.snd - y.snd :=
  rfl


@[simp]
theorem neg_fst : (-x).fst = -x.fst :=
  rfl


@[simp]
theorem neg_snd : (-x).snd = -x.snd :=
  rfl


@[simp]
theorem smul_fst : (c • x).fst = c • x.fst :=
  rfl


@[simp]
theorem smul_snd : (c • x).snd = c • x.snd :=
  rfl


@[simp]
theorem equiv_fst (x : WithLp p (α × β)) : (WithLp.equiv p (α × β) x).fst = x.fst :=
  rfl


@[simp]
theorem equiv_snd (x : WithLp p (α × β)) : (WithLp.equiv p (α × β) x).snd = x.snd :=
  rfl


@[simp]
theorem equiv_symm_fst (x : α × β) : ((WithLp.equiv p (α × β)).symm x).fst = x.fst :=
  rfl


@[simp]
theorem equiv_symm_snd (x : α × β) : ((WithLp.equiv p (α × β)).symm x).snd = x.snd :=
  rfl


open scoped Classical in
/-- Endowing the space `WithLp p (α × β)` with the `L^p` edistance. We register this instance
separate from `WithLp.instProdPseudoEMetric` since the latter requires the type class hypothesis
`[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future emetric-like structure on `WithLp p (α × β)` for
`p < 1` satisfying a relaxed triangle inequality. The terminology for this varies throughout the
literature, but it is sometimes called a *quasi-metric* or *semi-metric*. -/
instance instProdEDist : EDist (WithLp p (α × β)) where
  edist f g :=
    if _hp : p = 0 then
      (if edist f.fst g.fst = 0 then 0 else 1) + (if edist f.snd g.snd = 0 then 0 else 1)
    else if p = ∞ then
      edist f.fst g.fst ⊔ edist f.snd g.snd
    else
      (edist f.fst g.fst ^ p.toReal + edist f.snd g.snd ^ p.toReal) ^ (1 / p.toReal)


@[simp]
theorem prod_edist_eq_card (f g : WithLp 0 (α × β)) :
    edist f g =
      (if edist f.fst g.fst = 0 then 0 else 1) + (if edist f.snd g.snd = 0 then 0 else 1) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : EDist α
    inst✝ : EDist β
    f g : WithLp 0 (Prod α β)
    ⊢ Eq (EDist.edist f g) (HAdd.hAdd (ite (Eq (EDist.edist f.1 g.1) 0) 0 1) (ite  …
  -/
  convert if_pos rfl
  /-
    🎉 no goals
  -/


theorem prod_edist_eq_add (hp : 0 < p.toReal) (f g : WithLp p (α × β)) :
    edist f g = (edist f.fst g.fst ^ p.toReal + edist f.snd g.snd ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


theorem prod_edist_eq_sup (f g : WithLp ∞ (α × β)) :
    edist f g = edist f.fst g.fst ⊔ edist f.snd g.snd := rfl


/-- The distance from one point to itself is always zero.

This holds independent of `p` and does not require `[Fact (1 ≤ p)]`. We keep it separate
from `WithLp.instProdPseudoEMetricSpace` so it can be used also for `p < 1`. -/
theorem prod_edist_self (f : WithLp p (α × β)) : edist f f = 0 := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : WithLp p (Prod α β)
    ⊢ Eq (EDist.edist f f) 0
  -/
  rcases p.trichotomy with (rfl | rfl | h)
  · classical
    simp
    /-
      case inr.inl
      α : Type u_2
      β : Type u_3
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f : WithLp Top.top (Prod α β)
      ⊢ Eq (EDist.edist f f) 0
    -/
  · simp [prod_edist_eq_sup]
    /-
      🎉 no goals
    -/
  · simp [prod_edist_eq_add h, ENNReal.zero_rpow_of_pos h,
      ENNReal.zero_rpow_of_pos (inv_pos.2 <| h)]


/-- The distance is symmetric.

This holds independent of `p` and does not require `[Fact (1 ≤ p)]`. We keep it separate
from `WithLp.instProdPseudoEMetricSpace` so it can be used also for `p < 1`. -/
theorem prod_edist_comm (f g : WithLp p (α × β)) : edist f g = edist g f := by
  classical
  rcases p.trichotomy with (rfl | rfl | h)
  · simp only [prod_edist_eq_card, edist_comm]
  · simp only [prod_edist_eq_sup, edist_comm]
  · simp only [prod_edist_eq_add h, edist_comm]


open scoped Classical in
/-- Endowing the space `WithLp p (α × β)` with the `L^p` distance. We register this instance
separate from `WithLp.instProdPseudoMetricSpace` since the latter requires the type class hypothesis
`[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future metric-like structure on `WithLp p (α × β)` for
`p < 1` satisfying a relaxed triangle inequality. The terminology for this varies throughout the
literature, but it is sometimes called a *quasi-metric* or *semi-metric*. -/
instance instProdDist : Dist (WithLp p (α × β)) where
  dist f g :=
    if _hp : p = 0 then
      (if dist f.fst g.fst = 0 then 0 else 1) + (if dist f.snd g.snd = 0 then 0 else 1)
    else if p = ∞ then
      dist f.fst g.fst ⊔ dist f.snd g.snd
    else
      (dist f.fst g.fst ^ p.toReal + dist f.snd g.snd ^ p.toReal) ^ (1 / p.toReal)


theorem prod_dist_eq_card (f g : WithLp 0 (α × β)) : dist f g =
    (if dist f.fst g.fst = 0 then 0 else 1) + (if dist f.snd g.snd = 0 then 0 else 1) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Dist α
    inst✝ : Dist β
    f g : WithLp 0 (Prod α β)
    ⊢ Eq (Dist.dist f g) (HAdd.hAdd (ite (Eq (Dist.dist f.1 g.1) 0) 0 1) (ite (Eq  …
  -/
  convert if_pos rfl
  /-
    🎉 no goals
  -/


theorem prod_dist_eq_add (hp : 0 < p.toReal) (f g : WithLp p (α × β)) :
    dist f g = (dist f.fst g.fst ^ p.toReal + dist f.snd g.snd ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


theorem prod_dist_eq_sup (f g : WithLp ∞ (α × β)) :
    dist f g = dist f.fst g.fst ⊔ dist f.snd g.snd := rfl


open scoped Classical in
/-- Endowing the space `WithLp p (α × β)` with the `L^p` norm. We register this instance
separate from `WithLp.instProdSeminormedAddCommGroup` since the latter requires the type class
hypothesis `[Fact (1 ≤ p)]` in order to prove the triangle inequality.

Registering this separately allows for a future norm-like structure on `WithLp p (α × β)` for
`p < 1` satisfying a relaxed triangle inequality. These are called *quasi-norms*. -/
instance instProdNorm : Norm (WithLp p (α × β)) where
  norm f :=
    if _hp : p = 0 then
      (if ‖f.fst‖ = 0 then 0 else 1) + (if ‖f.snd‖ = 0 then 0 else 1)
    else if p = ∞ then
      ‖f.fst‖ ⊔ ‖f.snd‖
    else
      (‖f.fst‖ ^ p.toReal + ‖f.snd‖ ^ p.toReal) ^ (1 / p.toReal)


@[simp]
theorem prod_norm_eq_card (f : WithLp 0 (α × β)) :
    ‖f‖ = (if ‖f.fst‖ = 0 then 0 else 1) + (if ‖f.snd‖ = 0 then 0 else 1) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : Norm α
    inst✝ : Norm β
    f : WithLp 0 (Prod α β)
    ⊢ Eq (Norm.norm f) (HAdd.hAdd (ite (Eq (Norm.norm f.1) 0) 0 1) (ite (Eq (Norm. …
  -/
  convert if_pos rfl
  /-
    🎉 no goals
  -/


theorem prod_norm_eq_sup (f : WithLp ∞ (α × β)) : ‖f‖ = ‖f.fst‖ ⊔ ‖f.snd‖ := rfl


theorem prod_norm_eq_add (hp : 0 < p.toReal) (f : WithLp p (α × β)) :
    ‖f‖ = (‖f.fst‖ ^ p.toReal + ‖f.snd‖ ^ p.toReal) ^ (1 / p.toReal) :=
  let hp' := ENNReal.toReal_pos_iff.mp hp
  (if_neg hp'.1.ne').trans (if_neg hp'.2.ne)


/-- Endowing the space `WithLp p (α × β)` with the `L^p` pseudoemetric structure. This definition is
not satisfactory, as it does not register the fact that the topology and the uniform structure
coincide with the product one. Therefore, we do not register it as an instance. Using this as a
temporary pseudoemetric space instance, we will show that the uniform structure is equal (but not
defeq) to the product one, and then register an instance in which we replace the uniform structure
by the product one using this pseudoemetric space and `PseudoEMetricSpace.replaceUniformity`. -/
def prodPseudoEMetricAux [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    PseudoEMetricSpace (WithLp p (α × β)) where
  edist_self := prod_edist_self p
  edist_comm := prod_edist_comm p
  edist_triangle f g h := by
    /-
      p : ENNReal
      𝕜 : Type u_1
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f g h : WithLp p (Prod α β)
      ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
    -/
    rcases p.dichotomy with (rfl | hp)
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : PseudoEMetricSpace β
        hp : Fact (LE.le 1 Top.top)
        f g h : WithLp Top.top (Prod α β)
        ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
      -/
    · simp only [prod_edist_eq_sup]
      exact sup_le ((edist_triangle _ g.fst _).trans <| add_le_add le_sup_left le_sup_left)
        ((edist_triangle _ g.snd _).trans <| add_le_add le_sup_right le_sup_right)
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        hp✝ : Fact (LE.le 1 p)
        inst✝¹ : PseudoEMetricSpace α
        inst✝ : PseudoEMetricSpace β
        f g h : WithLp p (Prod α β)
        hp : LE.le 1 p.toReal
        ⊢ LE.le (EDist.edist f h) (HAdd.hAdd (EDist.edist f g) (EDist.edist g h))
      -/
    · simp only [prod_edist_eq_add (zero_lt_one.trans_le hp)]
      calc
        (edist f.fst h.fst ^ p.toReal + edist f.snd h.snd ^ p.toReal) ^ (1 / p.toReal) ≤
            ((edist f.fst g.fst + edist g.fst h.fst) ^ p.toReal +
              (edist f.snd g.snd + edist g.snd h.snd) ^ p.toReal) ^ (1 / p.toReal) := by
          gcongr <;> apply edist_triangle
        _ ≤
            (edist f.fst g.fst ^ p.toReal + edist f.snd g.snd ^ p.toReal) ^ (1 / p.toReal) +
              (edist g.fst h.fst ^ p.toReal + edist g.snd h.snd ^ p.toReal) ^ (1 / p.toReal) := by
          have := ENNReal.Lp_add_le {0, 1}
            (if · = 0 then edist f.fst g.fst else edist f.snd g.snd)
            (if · = 0 then edist g.fst h.fst else edist g.snd h.snd) hp
          simp only [Finset.mem_singleton, not_false_eq_true, Finset.sum_insert,
            Finset.sum_singleton, reduceCtorEq] at this
          exact this


/-- An auxiliary lemma used twice in the proof of `WithLp.prodPseudoMetricAux` below. Not intended
for use outside this file. -/
theorem prod_sup_edist_ne_top_aux [PseudoMetricSpace α] [PseudoMetricSpace β]
    (f g : WithLp ∞ (α × β)) :
    edist f.fst g.fst ⊔ edist f.snd g.snd ≠ ⊤ :=
                 /-
                   α : Type u_2
                   β : Type u_3
                   inst✝¹ : PseudoMetricSpace α
                   inst✝ : PseudoMetricSpace β
                   f g : WithLp Top.top (Prod α β)
                   ⊢ LT.lt (Max.max (EDist.edist f.1 g.1) (EDist.edist f.2 g.2)) Top.top
                 -/
  ne_of_lt <| by simp [edist, PseudoMetricSpace.edist_dist]
                 /-
                   🎉 no goals
                 -/


/-- Endowing the space `WithLp p (α × β)` with the `L^p` pseudometric structure. This definition is
not satisfactory, as it does not register the fact that the topology, the uniform structure, and the
bornology coincide with the product ones. Therefore, we do not register it as an instance. Using
this as a temporary pseudoemetric space instance, we will show that the uniform structure is equal
(but not defeq) to the product one, and then register an instance in which we replace the uniform
structure and the bornology by the product ones using this pseudometric space,
`PseudoMetricSpace.replaceUniformity`, and `PseudoMetricSpace.replaceBornology`.

See note [reducible non-instances] -/
abbrev prodPseudoMetricAux [PseudoMetricSpace α] [PseudoMetricSpace β] :
    PseudoMetricSpace (WithLp p (α × β)) :=
  PseudoEMetricSpace.toPseudoMetricSpaceOfDist dist
    (fun f g => by
      /-
        p : ENNReal
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        hp : Fact (LE.le 1 p)
        inst✝¹ : PseudoMetricSpace α
        inst✝ : PseudoMetricSpace β
        f g : WithLp p (Prod α β)
        ⊢ Ne (EDist.edist f g) Top.top
      -/
      rcases p.dichotomy with (rfl | h)
        /-
          case inl
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          hp : Fact (LE.le 1 Top.top)
          f g : WithLp Top.top (Prod α β)
          ⊢ Ne (EDist.edist f g) Top.top
        -/
      · exact prod_sup_edist_ne_top_aux f g
        /-
          🎉 no goals
        -/
        /-
          case inr
          p : ENNReal
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          hp : Fact (LE.le 1 p)
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          f g : WithLp p (Prod α β)
          h : LE.le 1 p.toReal
          ⊢ Ne (EDist.edist f g) Top.top
        -/
      · rw [prod_edist_eq_add (zero_lt_one.trans_le h)]
        /-
          case inr
          p : ENNReal
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          hp : Fact (LE.le 1 p)
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          f g : WithLp p (Prod α β)
          h : LE.le 1 p.toReal
          ⊢ Ne (HPow.hPow (HAdd.hAdd (HPow.hPow (EDist.edist f.1 g.1) p.toReal) (HPow.hP …
        -/
        refine ENNReal.rpow_ne_top_of_nonneg (by positivity) (ne_of_lt ?_)
        /-
          case inr
          p : ENNReal
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          hp : Fact (LE.le 1 p)
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          f g : WithLp p (Prod α β)
          h : LE.le 1 p.toReal
          ⊢ LT.lt (HAdd.hAdd (HPow.hPow (EDist.edist f.1 g.1) p.toReal) (HPow.hPow (EDis …
        -/
        simp [ENNReal.add_lt_top, ENNReal.rpow_lt_top_of_nonneg, edist_ne_top] )
        /-
          🎉 no goals
        -/
    fun f g => by
    /-
      p : ENNReal
      𝕜 : Type u_1
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      f g : WithLp p (Prod α β)
      ⊢ Eq (Dist.dist f g) (EDist.edist f g).toReal
    -/
    rcases p.dichotomy with (rfl | h)
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : PseudoMetricSpace α
        inst✝ : PseudoMetricSpace β
        hp : Fact (LE.le 1 Top.top)
        f g : WithLp Top.top (Prod α β)
        ⊢ Eq (Dist.dist f g) (EDist.edist f g).toReal
      -/
    · rw [prod_edist_eq_sup, prod_dist_eq_sup]
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : PseudoMetricSpace α
        inst✝ : PseudoMetricSpace β
        hp : Fact (LE.le 1 Top.top)
        f g : WithLp Top.top (Prod α β)
        ⊢ Eq (Max.max (Dist.dist f.1 g.1) (Dist.dist f.2 g.2)) (Max.max (EDist.edist f …
      -/
      refine le_antisymm (sup_le ?_ ?_) ?_
      · rw [← ENNReal.ofReal_le_iff_le_toReal (prod_sup_edist_ne_top_aux f g),
          ← PseudoMetricSpace.edist_dist]
        /-
          case inl.refine_1
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          hp : Fact (LE.le 1 Top.top)
          f g : WithLp Top.top (Prod α β)
          ⊢ LE.le (PseudoMetricSpace.edist f.1 g.1) (Max.max (EDist.edist f.1 g.1) (EDis …
        -/
        exact le_sup_left
        /-
          🎉 no goals
        -/
      · rw [← ENNReal.ofReal_le_iff_le_toReal (prod_sup_edist_ne_top_aux f g),
          ← PseudoMetricSpace.edist_dist]
        /-
          case inl.refine_2
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          hp : Fact (LE.le 1 Top.top)
          f g : WithLp Top.top (Prod α β)
          ⊢ LE.le (PseudoMetricSpace.edist f.2 g.2) (Max.max (EDist.edist f.1 g.1) (EDis …
        -/
        exact le_sup_right
        /-
          🎉 no goals
        -/
        /-
          case inl.refine_3
          𝕜 : Type u_1
          α : Type u_2
          β : Type u_3
          inst✝¹ : PseudoMetricSpace α
          inst✝ : PseudoMetricSpace β
          hp : Fact (LE.le 1 Top.top)
          f g : WithLp Top.top (Prod α β)
          ⊢ LE.le (Max.max (EDist.edist f.1 g.1) (EDist.edist f.2 g.2)).toReal (Max.max  …
        -/
      · refine ENNReal.toReal_le_of_le_ofReal ?_ ?_
          /-
            case inl.refine_3.refine_1
            𝕜 : Type u_1
            α : Type u_2
            β : Type u_3
            inst✝¹ : PseudoMetricSpace α
            inst✝ : PseudoMetricSpace β
            hp : Fact (LE.le 1 Top.top)
            f g : WithLp Top.top (Prod α β)
            ⊢ LE.le 0 (Max.max (Dist.dist f.1 g.1) (Dist.dist f.2 g.2))
          -/
        · simp only [le_sup_iff, dist_nonneg, or_self]
          /-
            🎉 no goals
          -/
          /-
            case inl.refine_3.refine_2
            𝕜 : Type u_1
            α : Type u_2
            β : Type u_3
            inst✝¹ : PseudoMetricSpace α
            inst✝ : PseudoMetricSpace β
            hp : Fact (LE.le 1 Top.top)
            f g : WithLp Top.top (Prod α β)
            ⊢ LE.le (Max.max (EDist.edist f.1 g.1) (EDist.edist f.2 g.2)) (ENNReal.ofReal  …
          -/
        · simp [edist, PseudoMetricSpace.edist_dist, ENNReal.ofReal_le_ofReal]
          /-
            🎉 no goals
          -/
    · have h1 : edist f.fst g.fst ^ p.toReal ≠ ⊤ :=
        ENNReal.rpow_ne_top_of_nonneg (zero_le_one.trans h) (edist_ne_top _ _)
      have h2 : edist f.snd g.snd ^ p.toReal ≠ ⊤ :=
        ENNReal.rpow_ne_top_of_nonneg (zero_le_one.trans h) (edist_ne_top _ _)
      simp only [prod_edist_eq_add (zero_lt_one.trans_le h), dist_edist, ENNReal.toReal_rpow,
        prod_dist_eq_add (zero_lt_one.trans_le h), ← ENNReal.toReal_add h1 h2]


theorem prod_lipschitzWith_equiv_aux [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    LipschitzWith 1 (WithLp.equiv p (α × β)) := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    ⊢ LipschitzWith 1 ⇑(WithLp.equiv p (Prod α β))
  -/
  intro x y
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    x y : WithLp p (Prod α β)
    ⊢ LE.le (EDist.edist ((WithLp.equiv p (Prod α β)) x) ((WithLp.equiv p (Prod α  …
  -/
  rcases p.dichotomy with (rfl | h)
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      hp : Fact (LE.le 1 Top.top)
      x y : WithLp Top.top (Prod α β)
      ⊢ LE.le (EDist.edist ((WithLp.equiv Top.top (Prod α β)) x) ((WithLp.equiv Top. …
    -/
  · simp [edist]
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      ⊢ LE.le (EDist.edist ((WithLp.equiv p (Prod α β)) x) ((WithLp.equiv p (Prod α  …
    -/
  · have cancel : p.toReal * (1 / p.toReal) = 1 := mul_div_cancel₀ 1 (zero_lt_one.trans_le h).ne'
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (EDist.edist ((WithLp.equiv p (Prod α β)) x) ((WithLp.equiv p (Prod α  …
    -/
    rw [prod_edist_eq_add (zero_lt_one.trans_le h)]
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (EDist.edist ((WithLp.equiv p (Prod α β)) x) ((WithLp.equiv p (Prod α  …
    -/
    simp only [edist, forall_prop_of_true, one_mul, ENNReal.coe_one, sup_le_iff]
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ And (LE.le (EDist.edist ((WithLp.equiv p (Prod α β)) x).1 ((WithLp.equiv p ( …
    -/
    constructor
    · calc
        edist x.fst y.fst ≤ (edist x.fst y.fst ^ p.toReal) ^ (1 / p.toReal) := by
          simp only [← ENNReal.rpow_mul, cancel, ENNReal.rpow_one, le_refl]
        _ ≤ (edist x.fst y.fst ^ p.toReal + edist x.snd y.snd ^ p.toReal) ^ (1 / p.toReal) := by
          gcongr
          simp only [self_le_add_right]
    · calc
        edist x.snd y.snd ≤ (edist x.snd y.snd ^ p.toReal) ^ (1 / p.toReal) := by
          simp only [← ENNReal.rpow_mul, cancel, ENNReal.rpow_one, le_refl]
        _ ≤ (edist x.fst y.fst ^ p.toReal + edist x.snd y.snd ^ p.toReal) ^ (1 / p.toReal) := by
          gcongr
          simp only [self_le_add_left]


theorem prod_antilipschitzWith_equiv_aux [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    AntilipschitzWith ((2 : ℝ≥0) ^ (1 / p).toReal) (WithLp.equiv p (α × β)) := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    ⊢ AntilipschitzWith (HPow.hPow 2 (HDiv.hDiv 1 p).toReal) ⇑(WithLp.equiv p (Pro …
  -/
  intro x y
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    x y : WithLp p (Prod α β)
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 p).toReal)) ( …
  -/
  rcases p.dichotomy with (rfl | h)
    /-
      case inl
      α : Type u_2
      β : Type u_3
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      hp : Fact (LE.le 1 Top.top)
      x y : WithLp Top.top (Prod α β)
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 Top.top).toRe …
    -/
  · simp [edist]
    /-
      🎉 no goals
    -/
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 p).toReal)) ( …
    -/
  · have pos : 0 < p.toReal := by positivity
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 p).toReal)) ( …
    -/
    have nonneg : 0 ≤ 1 / p.toReal := by positivity
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 p).toReal)) ( …
    -/
    have cancel : p.toReal * (1 / p.toReal) = 1 := mul_div_cancel₀ 1 (ne_of_gt pos)
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(HPow.hPow 2 (HDiv.hDiv 1 p).toReal)) ( …
    -/
    rw [prod_edist_eq_add pos, ENNReal.toReal_div 1 p]
    /-
      case inr
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      x y : WithLp p (Prod α β)
      h : LE.le 1 p.toReal
      pos : LT.lt 0 p.toReal
      nonneg : LE.le 0 (HDiv.hDiv 1 p.toReal)
      cancel : Eq (HMul.hMul p.toReal (HDiv.hDiv 1 p.toReal)) 1
      ⊢ LE.le (HPow.hPow (HAdd.hAdd (HPow.hPow (EDist.edist x.1 y.1) p.toReal) (HPow …
    -/
    simp only [edist, ← one_div, ENNReal.one_toReal]
    calc
      (edist x.fst y.fst ^ p.toReal + edist x.snd y.snd ^ p.toReal) ^ (1 / p.toReal) ≤
          (edist (WithLp.equiv p _ x) (WithLp.equiv p _ y) ^ p.toReal +
          edist (WithLp.equiv p _ x) (WithLp.equiv p _ y) ^ p.toReal) ^ (1 / p.toReal) := by
        gcongr <;> simp [edist]
      _ = (2 ^ (1 / p.toReal) : ℝ≥0) * edist (WithLp.equiv p _ x) (WithLp.equiv p _ y) := by
        simp only [← two_mul, ENNReal.mul_rpow_of_nonneg _ _ nonneg, ← ENNReal.rpow_mul, cancel,
          ENNReal.rpow_one, ENNReal.coe_rpow_of_nonneg _ nonneg, coe_ofNat]


theorem prod_aux_uniformity_eq [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    𝓤 (WithLp p (α × β)) = 𝓤[instUniformSpaceProd] := by
  have A : IsUniformInducing (WithLp.equiv p (α × β)) :=
    (prod_antilipschitzWith_equiv_aux p α β).isUniformInducing
      (prod_lipschitzWith_equiv_aux p α β).uniformContinuous
  have : (fun x : WithLp p (α × β) × WithLp p (α × β) =>
    ((WithLp.equiv p (α × β)) x.fst, (WithLp.equiv p (α × β)) x.snd)) = id := by
    ext i <;> rfl
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    A : IsUniformInducing ⇑(WithLp.equiv p (Prod α β))
    this : Eq (fun x => { fst := (WithLp.equiv p (Prod α β)) x.1, snd := (WithLp.e …
    ⊢ Eq (uniformity (WithLp p (Prod α β))) (uniformity (Prod α β))
  -/
  rw [← A.comap_uniformity, this, comap_id]
  /-
    🎉 no goals
  -/


theorem prod_aux_cobounded_eq [PseudoMetricSpace α] [PseudoMetricSpace β] :
    cobounded (WithLp p (α × β)) = @cobounded _ Prod.instBornology :=
  calc
    cobounded (WithLp p (α × β)) = comap (WithLp.equiv p (α × β)) (cobounded _) :=
      le_antisymm (prod_antilipschitzWith_equiv_aux p α β).tendsto_cobounded.le_comap
        (prod_lipschitzWith_equiv_aux p α β).comap_cobounded_le
    _ = _ := comap_id


instance instProdTopologicalSpace : TopologicalSpace (WithLp p (α × β)) :=
  instTopologicalSpaceProd


@[continuity]
theorem prod_continuous_equiv : Continuous (WithLp.equiv p (α × β)) :=
  continuous_id


@[continuity]
theorem prod_continuous_equiv_symm : Continuous (WithLp.equiv p (α × β)).symm :=
  continuous_id


instance instProdT0Space : T0Space (WithLp p (α × β)) :=
  Prod.instT0Space


instance instProdUniformSpace : UniformSpace (WithLp p (α × β)) :=
  instUniformSpaceProd


theorem prod_uniformContinuous_equiv : UniformContinuous (WithLp.equiv p (α × β)) :=
  uniformContinuous_id


theorem prod_uniformContinuous_equiv_symm : UniformContinuous (WithLp.equiv p (α × β)).symm :=
  uniformContinuous_id


instance instProdCompleteSpace : CompleteSpace (WithLp p (α × β)) :=
  CompleteSpace.prod


instance instProdBornology [Bornology α] [Bornology β] : Bornology (WithLp p (α × β)) :=
  Prod.instBornology


/-- `WithLp.equiv` as a continuous linear equivalence. -/
@[simps! (config := .asFn) apply symm_apply]
protected def prodContinuousLinearEquiv : WithLp p (α × β) ≃L[𝕜] α × β where
  toLinearEquiv := WithLp.linearEquiv _ _ _
  continuous_toFun := prod_continuous_equiv _ _ _
  continuous_invFun := prod_continuous_equiv_symm _ _ _


/-- `PseudoEMetricSpace` instance on the product of two pseudoemetric spaces, using the
`L^p` pseudoedistance, and having as uniformity the product uniformity. -/
instance instProdPseudoEMetricSpace [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    PseudoEMetricSpace (WithLp p (α × β)) :=
  (prodPseudoEMetricAux p α β).replaceUniformity (prod_aux_uniformity_eq p α β).symm


/-- `EMetricSpace` instance on the product of two emetric spaces, using the `L^p`
edistance, and having as uniformity the product uniformity. -/
instance instProdEMetricSpace [EMetricSpace α] [EMetricSpace β] : EMetricSpace (WithLp p (α × β)) :=
  EMetricSpace.ofT0PseudoEMetricSpace (WithLp p (α × β))


/-- `PseudoMetricSpace` instance on the product of two pseudometric spaces, using the
`L^p` distance, and having as uniformity the product uniformity. -/
instance instProdPseudoMetricSpace [PseudoMetricSpace α] [PseudoMetricSpace β] :
    PseudoMetricSpace (WithLp p (α × β)) :=
  ((prodPseudoMetricAux p α β).replaceUniformity
    (prod_aux_uniformity_eq p α β).symm).replaceBornology
    fun s => Filter.ext_iff.1 (prod_aux_cobounded_eq p α β).symm sᶜ


/-- `MetricSpace` instance on the product of two metric spaces, using the `L^p` distance,
and having as uniformity the product uniformity. -/
instance instProdMetricSpace [MetricSpace α] [MetricSpace β] : MetricSpace (WithLp p (α × β)) :=
  MetricSpace.ofT0PseudoMetricSpace _


theorem prod_nndist_eq_add [PseudoMetricSpace α] [PseudoMetricSpace β]
    (hp : p ≠ ∞) (x y : WithLp p (α × β)) :
    nndist x y = (nndist x.fst y.fst ^ p.toReal + nndist x.snd y.snd ^ p.toReal) ^ (1 / p.toReal) :=
  NNReal.eq <| by
    /-
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp✝ : Fact (LE.le 1 p)
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      hp : Ne p Top.top
      x y : WithLp p (Prod α β)
      ⊢ Eq ↑(NNDist.nndist x y) ↑(HPow.hPow (HAdd.hAdd (HPow.hPow (NNDist.nndist x.1 …
    -/
    push_cast
    /-
      p : ENNReal
      α : Type u_2
      β : Type u_3
      hp✝ : Fact (LE.le 1 p)
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      hp : Ne p Top.top
      x y : WithLp p (Prod α β)
      ⊢ Eq (Dist.dist x y) (HPow.hPow (HAdd.hAdd (HPow.hPow (Dist.dist x.1 y.1) p.to …
    -/
    exact prod_dist_eq_add (p.toReal_pos_iff_ne_top.mpr hp) _ _
    /-
      🎉 no goals
    -/


theorem prod_nndist_eq_sup [PseudoMetricSpace α] [PseudoMetricSpace β] (x y : WithLp ∞ (α × β)) :
    nndist x y = nndist x.fst y.fst ⊔ nndist x.snd y.snd :=
  NNReal.eq <| by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x y : WithLp Top.top (Prod α β)
      ⊢ Eq ↑(NNDist.nndist x y) ↑(Max.max (NNDist.nndist x.1 y.1) (NNDist.nndist x.2 …
    -/
    push_cast
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      x y : WithLp Top.top (Prod α β)
      ⊢ Eq (Dist.dist x y) (Max.max (Dist.dist x.1 y.1) (Dist.dist x.2 y.2))
    -/
    exact prod_dist_eq_sup _ _
    /-
      🎉 no goals
    -/


theorem prod_lipschitzWith_equiv [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    LipschitzWith 1 (WithLp.equiv p (α × β)) :=
  prod_lipschitzWith_equiv_aux p α β


theorem prod_antilipschitzWith_equiv [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    AntilipschitzWith ((2 : ℝ≥0) ^ (1 / p).toReal) (WithLp.equiv p (α × β)) :=
  prod_antilipschitzWith_equiv_aux p α β


theorem prod_infty_equiv_isometry [PseudoEMetricSpace α] [PseudoEMetricSpace β] :
    Isometry (WithLp.equiv ∞ (α × β)) :=
  fun x y =>
                  /-
                    α : Type u_2
                    β : Type u_3
                    inst✝¹ : PseudoEMetricSpace α
                    inst✝ : PseudoEMetricSpace β
                    x y : WithLp Top.top (Prod α β)
                    ⊢ LE.le (EDist.edist ((WithLp.equiv Top.top (Prod α β)) x) ((WithLp.equiv Top. …
                  -/
  le_antisymm (by simpa only [ENNReal.coe_one, one_mul] using prod_lipschitzWith_equiv ∞ α β x y)
                  /-
                    🎉 no goals
                  -/
    (by
      simpa only [ENNReal.div_top, ENNReal.zero_toReal, NNReal.rpow_zero, ENNReal.coe_one,
        one_mul] using prod_antilipschitzWith_equiv ∞ α β x y)


/-- Seminormed group instance on the product of two normed groups, using the `L^p`
norm. -/
instance instProdSeminormedAddCommGroup [SeminormedAddCommGroup α] [SeminormedAddCommGroup β] :
    SeminormedAddCommGroup (WithLp p (α × β)) where
  dist_eq x y := by
    /-
      p : ENNReal
      𝕜 : Type u_1
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x y : WithLp p (Prod α β)
      ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
    -/
    rcases p.dichotomy with (rfl | h)
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : SeminormedAddCommGroup β
        hp : Fact (LE.le 1 Top.top)
        x y : WithLp Top.top (Prod α β)
        ⊢ Eq (Dist.dist x y) (Norm.norm (HSub.hSub x y))
      -/
    · simp only [prod_dist_eq_sup, prod_norm_eq_sup, dist_eq_norm]
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : SeminormedAddCommGroup β
        hp : Fact (LE.le 1 Top.top)
        x y : WithLp Top.top (Prod α β)
        ⊢ Eq (Max.max (Norm.norm (HSub.hSub x.1 y.1)) (Norm.norm (HSub.hSub x.2 y.2))) …
      -/
      rfl
      /-
        🎉 no goals
      -/
    · simp only [prod_dist_eq_add (zero_lt_one.trans_le h),
        prod_norm_eq_add (zero_lt_one.trans_le h), dist_eq_norm]
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        hp : Fact (LE.le 1 p)
        inst✝¹ : SeminormedAddCommGroup α
        inst✝ : SeminormedAddCommGroup β
        x y : WithLp p (Prod α β)
        h : LE.le 1 p.toReal
        ⊢ Eq (HPow.hPow (HAdd.hAdd (HPow.hPow (Norm.norm (HSub.hSub x.1 y.1)) p.toReal …
      -/
      rfl
      /-
        🎉 no goals
      -/


/-- normed group instance on the product of two normed groups, using the `L^p` norm. -/
instance instProdNormedAddCommGroup [NormedAddCommGroup α] [NormedAddCommGroup β] :
    NormedAddCommGroup (WithLp p (α × β)) :=
  { instProdSeminormedAddCommGroup p α β with
    eq_of_dist_eq_zero := eq_of_dist_eq_zero }


theorem prod_norm_eq_of_nat [Norm α] [Norm β] (n : ℕ) (h : p = n) (f : WithLp p (α × β)) :
    ‖f‖ = (‖f.fst‖ ^ n + ‖f.snd‖ ^ n) ^ (1 / (n : ℝ)) := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : Norm α
    inst✝ : Norm β
    n : Nat
    h : Eq p ↑n
    f : WithLp p (Prod α β)
    ⊢ Eq (Norm.norm f) (HPow.hPow (HAdd.hAdd (HPow.hPow (Norm.norm f.1) n) (HPow.h …
  -/
  have := p.toReal_pos_iff_ne_top.mpr (ne_of_eq_of_ne h <| ENNReal.natCast_ne_top n)
  simp only [one_div, h, Real.rpow_natCast, ENNReal.toReal_nat, eq_self_iff_true, Finset.sum_congr,
    prod_norm_eq_add this]


theorem prod_nnnorm_eq_add (hp : p ≠ ∞) (f : WithLp p (α × β)) :
    ‖f‖₊ = (‖f.fst‖₊ ^ p.toReal + ‖f.snd‖₊ ^ p.toReal) ^ (1 / p.toReal) := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp✝ : Fact (LE.le 1 p)
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    hp : Ne p Top.top
    f : WithLp p (Prod α β)
    ⊢ Eq (NNNorm.nnnorm f) (HPow.hPow (HAdd.hAdd (HPow.hPow (NNNorm.nnnorm f.1) p. …
  -/
  ext
  /-
    case a
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp✝ : Fact (LE.le 1 p)
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    hp : Ne p Top.top
    f : WithLp p (Prod α β)
    ⊢ Eq ↑(NNNorm.nnnorm f) ↑(HPow.hPow (HAdd.hAdd (HPow.hPow (NNNorm.nnnorm f.1)  …
  -/
  simp [prod_norm_eq_add (p.toReal_pos_iff_ne_top.mpr hp)]
  /-
    🎉 no goals
  -/


theorem prod_nnnorm_eq_sup (f : WithLp ∞ (α × β)) : ‖f‖₊ = ‖f.fst‖₊ ⊔  ‖f.snd‖₊ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    f : WithLp Top.top (Prod α β)
    ⊢ Eq (NNNorm.nnnorm f) (Max.max (NNNorm.nnnorm f.1) (NNNorm.nnnorm f.2))
  -/
  ext
  /-
    case a
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    f : WithLp Top.top (Prod α β)
    ⊢ Eq ↑(NNNorm.nnnorm f) ↑(Max.max (NNNorm.nnnorm f.1) (NNNorm.nnnorm f.2))
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp] theorem prod_nnnorm_equiv (f : WithLp ∞ (α × β)) : ‖WithLp.equiv ⊤ _ f‖₊ = ‖f‖₊ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    f : WithLp Top.top (Prod α β)
    ⊢ Eq (NNNorm.nnnorm ((WithLp.equiv Top.top (Prod α β)) f)) (NNNorm.nnnorm f)
  -/
  rw [prod_nnnorm_eq_sup, Prod.nnnorm_def', equiv_fst, equiv_snd]
  /-
    🎉 no goals
  -/


@[simp] theorem prod_nnnorm_equiv_symm (f : α × β) : ‖(WithLp.equiv ⊤ _).symm f‖₊ = ‖f‖₊ :=
  (prod_nnnorm_equiv _).symm


@[simp] theorem prod_norm_equiv (f : WithLp ∞ (α × β)) : ‖WithLp.equiv ⊤ _ f‖ = ‖f‖ :=
  congr_arg NNReal.toReal <| prod_nnnorm_equiv f


@[simp] theorem prod_norm_equiv_symm (f : α × β) : ‖(WithLp.equiv ⊤ _).symm f‖ = ‖f‖ :=
  (prod_norm_equiv _).symm


theorem prod_norm_eq_of_L1 (x : WithLp 1 (α × β)) :
    ‖x‖ = ‖x.fst‖ + ‖x.snd‖ := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x : WithLp 1 (Prod α β)
    ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2))
  -/
  simp [prod_norm_eq_add]
  /-
    🎉 no goals
  -/


theorem prod_nnnorm_eq_of_L1 (x : WithLp 1 (α × β)) :
    ‖x‖₊ = ‖x.fst‖₊ + ‖x.snd‖₊ :=
  NNReal.eq <| by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x : WithLp 1 (Prod α β)
      ⊢ Eq ↑(NNNorm.nnnorm x) ↑(HAdd.hAdd (NNNorm.nnnorm x.1) (NNNorm.nnnorm x.2))
    -/
    push_cast
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x : WithLp 1 (Prod α β)
      ⊢ Eq (Norm.norm x) (HAdd.hAdd (Norm.norm x.1) (Norm.norm x.2))
    -/
    exact prod_norm_eq_of_L1 x
    /-
      🎉 no goals
    -/


theorem prod_dist_eq_of_L1 (x y : WithLp 1 (α × β)) :
    dist x y = dist x.fst y.fst + dist x.snd y.snd := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x y : WithLp 1 (Prod α β)
    ⊢ Eq (Dist.dist x y) (HAdd.hAdd (Dist.dist x.1 y.1) (Dist.dist x.2 y.2))
  -/
  simp_rw [dist_eq_norm, prod_norm_eq_of_L1, sub_fst, sub_snd]
  /-
    🎉 no goals
  -/


theorem prod_nndist_eq_of_L1 (x y : WithLp 1 (α × β)) :
    nndist x y = nndist x.fst y.fst + nndist x.snd y.snd :=
  NNReal.eq <| by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x y : WithLp 1 (Prod α β)
      ⊢ Eq ↑(NNDist.nndist x y) ↑(HAdd.hAdd (NNDist.nndist x.1 y.1) (NNDist.nndist x …
    -/
    push_cast
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x y : WithLp 1 (Prod α β)
      ⊢ Eq (Dist.dist x y) (HAdd.hAdd (Dist.dist x.1 y.1) (Dist.dist x.2 y.2))
    -/
    exact prod_dist_eq_of_L1 _ _
    /-
      🎉 no goals
    -/


theorem prod_edist_eq_of_L1 (x y : WithLp 1 (α × β)) :
    edist x y = edist x.fst y.fst + edist x.snd y.snd := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x y : WithLp 1 (Prod α β)
    ⊢ Eq (EDist.edist x y) (HAdd.hAdd (EDist.edist x.1 y.1) (EDist.edist x.2 y.2))
  -/
  simp [prod_edist_eq_add]
  /-
    🎉 no goals
  -/


theorem prod_norm_eq_of_L2 (x : WithLp 2 (α × β)) :
    ‖x‖ = √(‖x.fst‖ ^ 2 + ‖x.snd‖ ^ 2) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x : WithLp 2 (Prod α β)
    ⊢ Eq (Norm.norm x) (HAdd.hAdd (HPow.hPow (Norm.norm x.1) 2) (HPow.hPow (Norm.n …
  -/
  rw [prod_norm_eq_of_nat 2 (by norm_cast) _, Real.sqrt_eq_rpow]
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x : WithLp 2 (Prod α β)
    ⊢ Eq (HPow.hPow (HAdd.hAdd (HPow.hPow (Norm.norm x.1) 2) (HPow.hPow (Norm.norm …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


theorem prod_nnnorm_eq_of_L2 (x : WithLp 2 (α × β)) :
    ‖x‖₊ = NNReal.sqrt (‖x.fst‖₊ ^ 2 + ‖x.snd‖₊ ^ 2) :=
  NNReal.eq <| by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x : WithLp 2 (Prod α β)
      ⊢ Eq ↑(NNNorm.nnnorm x) ↑(NNReal.sqrt (HAdd.hAdd (HPow.hPow (NNNorm.nnnorm x.1 …
    -/
    push_cast
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x : WithLp 2 (Prod α β)
      ⊢ Eq (Norm.norm x) (HAdd.hAdd (HPow.hPow (Norm.norm x.1) 2) (HPow.hPow (Norm.n …
    -/
    exact prod_norm_eq_of_L2 x
    /-
      🎉 no goals
    -/


theorem prod_norm_sq_eq_of_L2 (x : WithLp 2 (α × β)) : ‖x‖ ^ 2 = ‖x.fst‖ ^ 2 + ‖x.snd‖ ^ 2 := by
  suffices ‖x‖₊ ^ 2 = ‖x.fst‖₊ ^ 2 + ‖x.snd‖₊ ^ 2 by
    simpa only [NNReal.coe_sum] using congr_arg ((↑) : ℝ≥0 → ℝ) this
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x : WithLp 2 (Prod α β)
    ⊢ Eq (HPow.hPow (NNNorm.nnnorm x) 2) (HAdd.hAdd (HPow.hPow (NNNorm.nnnorm x.1) …
  -/
  rw [prod_nnnorm_eq_of_L2, NNReal.sq_sqrt]
  /-
    🎉 no goals
  -/


theorem prod_dist_eq_of_L2 (x y : WithLp 2 (α × β)) :
    dist x y = √(dist x.fst y.fst ^ 2 + dist x.snd y.snd ^ 2) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x y : WithLp 2 (Prod α β)
    ⊢ Eq (Dist.dist x y) (HAdd.hAdd (HPow.hPow (Dist.dist x.1 y.1) 2) (HPow.hPow ( …
  -/
  simp_rw [dist_eq_norm, prod_norm_eq_of_L2, sub_fst, sub_snd]
  /-
    🎉 no goals
  -/


theorem prod_nndist_eq_of_L2 (x y : WithLp 2 (α × β)) :
    nndist x y = NNReal.sqrt (nndist x.fst y.fst ^ 2 + nndist x.snd y.snd ^ 2) :=
  NNReal.eq <| by
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x y : WithLp 2 (Prod α β)
      ⊢ Eq ↑(NNDist.nndist x y) ↑(NNReal.sqrt (HAdd.hAdd (HPow.hPow (NNDist.nndist x …
    -/
    push_cast
    /-
      α : Type u_2
      β : Type u_3
      inst✝¹ : SeminormedAddCommGroup α
      inst✝ : SeminormedAddCommGroup β
      x y : WithLp 2 (Prod α β)
      ⊢ Eq (Dist.dist x y) (HAdd.hAdd (HPow.hPow (Dist.dist x.1 y.1) 2) (HPow.hPow ( …
    -/
    exact prod_dist_eq_of_L2 _ _
    /-
      🎉 no goals
    -/


theorem prod_edist_eq_of_L2 (x y : WithLp 2 (α × β)) :
    edist x y = (edist x.fst y.fst ^ 2 + edist x.snd y.snd ^ 2) ^ (1 / 2 : ℝ) := by
  /-
    α : Type u_2
    β : Type u_3
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x y : WithLp 2 (Prod α β)
    ⊢ Eq (EDist.edist x y) (HPow.hPow (HAdd.hAdd (HPow.hPow (EDist.edist x.1 y.1)  …
  -/
  simp [prod_edist_eq_add]
  /-
    🎉 no goals
  -/


@[simp]
theorem nnnorm_equiv_symm_fst (x : α) :
    ‖(WithLp.equiv p (α × β)).symm (x, 0)‖₊ = ‖x‖₊ := by
  induction p generalizing hp with
  | top =>
    simp [prod_nnnorm_eq_sup]
  | coe p =>
    have hp0 : (p : ℝ) ≠ 0 := mod_cast (zero_lt_one.trans_le <| Fact.out (p := 1 ≤ (p : ℝ≥0∞))).ne'
    simp [prod_nnnorm_eq_add, NNReal.zero_rpow hp0, ← NNReal.rpow_mul, mul_inv_cancel₀ hp0]


@[simp]
theorem nnnorm_equiv_symm_snd (y : β) :
    ‖(WithLp.equiv p (α × β)).symm (0, y)‖₊ = ‖y‖₊ := by
  induction p generalizing hp with
  | top =>
    simp [prod_nnnorm_eq_sup]
  | coe p =>
    have hp0 : (p : ℝ) ≠ 0 := mod_cast (zero_lt_one.trans_le <| Fact.out (p := 1 ≤ (p : ℝ≥0∞))).ne'
    simp [prod_nnnorm_eq_add, NNReal.zero_rpow hp0, ← NNReal.rpow_mul, mul_inv_cancel₀ hp0]


@[simp]
theorem norm_equiv_symm_fst (x : α) : ‖(WithLp.equiv p (α × β)).symm (x, 0)‖ = ‖x‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_equiv_symm_fst p α β x


@[simp]
theorem norm_equiv_symm_snd (y : β) : ‖(WithLp.equiv p (α × β)).symm (0, y)‖ = ‖y‖ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nnnorm_equiv_symm_snd p α β y


@[simp]
theorem nndist_equiv_symm_fst (x₁ x₂ : α) :
    nndist ((WithLp.equiv p (α × β)).symm (x₁, 0)) ((WithLp.equiv p (α × β)).symm (x₂, 0)) =
      nndist x₁ x₂ := by
  rw [nndist_eq_nnnorm, nndist_eq_nnnorm, ← WithLp.equiv_symm_sub, Prod.mk_sub_mk, sub_zero,
    nnnorm_equiv_symm_fst]


@[simp]
theorem nndist_equiv_symm_snd (y₁ y₂ : β) :
    nndist ((WithLp.equiv p (α × β)).symm (0, y₁)) ((WithLp.equiv p (α × β)).symm (0, y₂)) =
      nndist y₁ y₂ := by
  rw [nndist_eq_nnnorm, nndist_eq_nnnorm, ← WithLp.equiv_symm_sub, Prod.mk_sub_mk, sub_zero,
    nnnorm_equiv_symm_snd]


@[simp]
theorem dist_equiv_symm_fst (x₁ x₂ : α) :
    dist ((WithLp.equiv p (α × β)).symm (x₁, 0)) ((WithLp.equiv p (α × β)).symm (x₂, 0)) =
      dist x₁ x₂ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nndist_equiv_symm_fst p α β x₁ x₂


@[simp]
theorem dist_equiv_symm_snd (y₁ y₂ : β) :
    dist ((WithLp.equiv p (α × β)).symm (0, y₁)) ((WithLp.equiv p (α × β)).symm (0, y₂)) =
      dist y₁ y₂ :=
  congr_arg ((↑) : ℝ≥0 → ℝ) <| nndist_equiv_symm_snd p α β y₁ y₂


@[simp]
theorem edist_equiv_symm_fst (x₁ x₂ : α) :
    edist ((WithLp.equiv p (α × β)).symm (x₁, 0)) ((WithLp.equiv p (α × β)).symm (x₂, 0)) =
      edist x₁ x₂ := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    x₁ x₂ : α
    ⊢ Eq (EDist.edist ((WithLp.equiv p (Prod α β)).symm { fst := x₁, snd := 0 }) ( …
  -/
  simp only [edist_nndist, nndist_equiv_symm_fst p α β x₁ x₂]
  /-
    🎉 no goals
  -/


@[simp]
theorem edist_equiv_symm_snd (y₁ y₂ : β) :
    edist ((WithLp.equiv p (α × β)).symm (0, y₁)) ((WithLp.equiv p (α × β)).symm (0, y₂)) =
      edist y₁ y₂ := by
  /-
    p : ENNReal
    α : Type u_2
    β : Type u_3
    hp : Fact (LE.le 1 p)
    inst✝¹ : SeminormedAddCommGroup α
    inst✝ : SeminormedAddCommGroup β
    y₁ y₂ : β
    ⊢ Eq (EDist.edist ((WithLp.equiv p (Prod α β)).symm { fst := 0, snd := y₁ }) ( …
  -/
  simp only [edist_nndist, nndist_equiv_symm_snd p α β y₁ y₂]
  /-
    🎉 no goals
  -/


instance instProdBoundedSMul : BoundedSMul 𝕜 (WithLp p (α × β)) :=
  .of_nnnorm_smul_le fun c f => by
    /-
      p : ENNReal
      𝕜 : Type u_1
      α : Type u_2
      β : Type u_3
      hp : Fact (LE.le 1 p)
      inst✝⁶ : SeminormedAddCommGroup α
      inst✝⁵ : SeminormedAddCommGroup β
      inst✝⁴ : SeminormedRing 𝕜
      inst✝³ : Module 𝕜 α
      inst✝² : Module 𝕜 β
      inst✝¹ : BoundedSMul 𝕜 α
      inst✝ : BoundedSMul 𝕜 β
      c : 𝕜
      f : WithLp p (Prod α β)
      ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
    -/
    rcases p.dichotomy with (rfl | hp)
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : SeminormedAddCommGroup α
        inst✝⁵ : SeminormedAddCommGroup β
        inst✝⁴ : SeminormedRing 𝕜
        inst✝³ : Module 𝕜 α
        inst✝² : Module 𝕜 β
        inst✝¹ : BoundedSMul 𝕜 α
        inst✝ : BoundedSMul 𝕜 β
        c : 𝕜
        hp : Fact (LE.le 1 Top.top)
        f : WithLp Top.top (Prod α β)
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
    · simp only [← prod_nnnorm_equiv, WithLp.equiv_smul]
      /-
        case inl
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        inst✝⁶ : SeminormedAddCommGroup α
        inst✝⁵ : SeminormedAddCommGroup β
        inst✝⁴ : SeminormedRing 𝕜
        inst✝³ : Module 𝕜 α
        inst✝² : Module 𝕜 β
        inst✝¹ : BoundedSMul 𝕜 α
        inst✝ : BoundedSMul 𝕜 β
        c : 𝕜
        hp : Fact (LE.le 1 Top.top)
        f : WithLp Top.top (Prod α β)
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c ((WithLp.equiv Top.top (Prod α β)) f)))  …
      -/
      exact norm_smul_le _ _
      /-
        🎉 no goals
      -/
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        hp✝ : Fact (LE.le 1 p)
        inst✝⁶ : SeminormedAddCommGroup α
        inst✝⁵ : SeminormedAddCommGroup β
        inst✝⁴ : SeminormedRing 𝕜
        inst✝³ : Module 𝕜 α
        inst✝² : Module 𝕜 β
        inst✝¹ : BoundedSMul 𝕜 α
        inst✝ : BoundedSMul 𝕜 β
        c : 𝕜
        f : WithLp p (Prod α β)
        hp : LE.le 1 p.toReal
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
    · have hp0 : 0 < p.toReal := zero_lt_one.trans_le hp
      /-
        case inr
        p : ENNReal
        𝕜 : Type u_1
        α : Type u_2
        β : Type u_3
        hp✝ : Fact (LE.le 1 p)
        inst✝⁶ : SeminormedAddCommGroup α
        inst✝⁵ : SeminormedAddCommGroup β
        inst✝⁴ : SeminormedRing 𝕜
        inst✝³ : Module 𝕜 α
        inst✝² : Module 𝕜 β
        inst✝¹ : BoundedSMul 𝕜 α
        inst✝ : BoundedSMul 𝕜 β
        c : 𝕜
        f : WithLp p (Prod α β)
        hp : LE.le 1 p.toReal
        hp0 : LT.lt 0 p.toReal
        ⊢ LE.le (NNNorm.nnnorm (HSMul.hSMul c f)) (HMul.hMul (NNNorm.nnnorm c) (NNNorm …
      -/
      have hpt : p ≠ ⊤ := p.toReal_pos_iff_ne_top.mp hp0
      rw [prod_nnnorm_eq_add hpt, prod_nnnorm_eq_add hpt, one_div, NNReal.rpow_inv_le_iff hp0,
        NNReal.mul_rpow, ← NNReal.rpow_mul, inv_mul_cancel₀ hp0.ne', NNReal.rpow_one, mul_add,
        ← NNReal.mul_rpow, ← NNReal.mul_rpow]
      exact add_le_add
        (NNReal.rpow_le_rpow (nnnorm_smul_le _ _) hp0.le)
        (NNReal.rpow_le_rpow (nnnorm_smul_le _ _) hp0.le)


/-- The canonical map `WithLp.equiv` between `WithLp ∞ (α × β)` and `α × β` as a linear isometric
equivalence. -/
def prodEquivₗᵢ : WithLp ∞ (α × β) ≃ₗᵢ[𝕜] α × β where
  __ := WithLp.equiv ∞ (α × β)
  map_add' _f _g := rfl
  map_smul' _c _f := rfl
  norm_map' := prod_norm_equiv


/-- The product of two normed spaces is a normed space, with the `L^p` norm. -/
instance instProdNormedSpace [NormedField 𝕜] [NormedSpace 𝕜 α] [NormedSpace 𝕜 β] :
    NormedSpace 𝕜 (WithLp p (α × β)) where
  norm_smul_le := norm_smul_le


