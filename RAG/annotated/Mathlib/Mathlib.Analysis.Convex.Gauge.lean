/-- The Minkowski functional. Given a set `s` in a real vector space, `gauge s` is the functional
which sends `x : E` to the smallest `r : ℝ` such that `x` is in `s` scaled by `r`. -/
def gauge (s : Set E) (x : E) : ℝ :=
  sInf { r : ℝ | 0 < r ∧ x ∈ r • s }


theorem gauge_def : gauge s x = sInf ({ r ∈ Set.Ioi (0 : ℝ) | x ∈ r • s }) :=
  rfl


/-- An alternative definition of the gauge using scalar multiplication on the element rather than on
the set. -/
theorem gauge_def' : gauge s x = sInf {r ∈ Set.Ioi (0 : ℝ) | r⁻¹ • x ∈ s} := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    ⊢ Eq (gauge s x) (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0)  …
  -/
  congrm sInf {r | ?_}
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    r : Real
    ⊢ Iff (And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r s) x)) (And (Membership. …
  -/
  exact and_congr_right fun hr => mem_smul_set_iff_inv_smul_mem₀ hr.ne' _ _
  /-
    🎉 no goals
  -/


private theorem gauge_set_bddBelow : BddBelow { r : ℝ | 0 < r ∧ x ∈ r • s } :=
  ⟨0, fun _ hr => hr.1.le⟩


/-- If the given subset is `Absorbent` then the set we take an infimum over in `gauge` is nonempty,
which is useful for proving many properties about the gauge. -/
theorem Absorbent.gauge_set_nonempty (absorbs : Absorbent ℝ s) :
    { r : ℝ | 0 < r ∧ x ∈ r • s }.Nonempty :=
  let ⟨r, hr₁, hr₂⟩ := (absorbs x).exists_pos
  ⟨r, hr₁, hr₂ r (Real.norm_of_nonneg hr₁.le).ge rfl⟩


theorem gauge_mono (hs : Absorbent ℝ s) (h : s ⊆ t) : gauge t ≤ gauge s := fun _ =>
  csInf_le_csInf gauge_set_bddBelow hs.gauge_set_nonempty fun _ hr => ⟨hr.1, smul_set_mono h hr.2⟩


theorem exists_lt_of_gauge_lt (absorbs : Absorbent ℝ s) (h : gauge s x < a) :
    ∃ b, 0 < b ∧ b < a ∧ x ∈ b • s := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    absorbs : Absorbent Real s
    h : LT.lt (gauge s x) a
    ⊢ Exists fun b => And (LT.lt 0 b) (And (LT.lt b a) (Membership.mem (HSMul.hSMu …
  -/
  obtain ⟨b, ⟨hb, hx⟩, hba⟩ := exists_lt_of_csInf_lt absorbs.gauge_set_nonempty h
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    absorbs : Absorbent Real s
    h : LT.lt (gauge s x) a
    b : Real
    hba : LT.lt b a
    hb : LT.lt 0 b
    hx : Membership.mem (HSMul.hSMul b s) x
    ⊢ Exists fun b => And (LT.lt 0 b) (And (LT.lt b a) (Membership.mem (HSMul.hSMu …
  -/
  exact ⟨b, hb, hba, hx⟩
  /-
    🎉 no goals
  -/


/-- The gauge evaluated at `0` is always zero (mathematically this requires `0` to be in the set `s`
but, the real infimum of the empty set in Lean being defined as `0`, it holds unconditionally). -/
@[simp]
theorem gauge_zero : gauge s 0 = 0 := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    ⊢ Eq (gauge s 0) 0
  -/
  rw [gauge_def']
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
  -/
  by_cases h : (0 : E) ∈ s
    /-
      case pos
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      h : Membership.mem s 0
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
    -/
  · simp only [smul_zero, sep_true, h, csInf_Ioi]
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      h : Not (Membership.mem s 0)
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
    -/
  · simp only [smul_zero, sep_false, h, Real.sInf_empty]
    /-
      🎉 no goals
    -/


@[simp]
theorem gauge_zero' : gauge (0 : Set E) = 0 := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    ⊢ Eq (gauge 0) 0
  -/
  ext x
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    x : E
    ⊢ Eq (gauge 0 x) (0 x)
  -/
  rw [gauge_def']
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    x : E
    ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
  -/
  obtain rfl | hx := eq_or_ne x 0
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
    -/
  · simp only [csInf_Ioi, mem_zero, Pi.zero_apply, eq_self_iff_true, sep_true, smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.inr
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      x : E
      hx : Ne x 0
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
    -/
  · simp only [mem_zero, Pi.zero_apply, inv_eq_zero, smul_eq_zero]
    /-
      case h.inr
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      x : E
      hx : Ne x 0
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r …
    -/
    convert Real.sInf_empty
    /-
      case h.e'_2.h.e'_3
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      x : E
      hx : Ne x 0
      ⊢ Eq (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r 0) (Eq x 0)) …
    -/
    exact eq_empty_iff_forall_not_mem.2 fun r hr => hr.2.elim (ne_of_gt hr.1) hx
    /-
      🎉 no goals
    -/


@[simp]
theorem gauge_empty : gauge (∅ : Set E) = 0 := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    ⊢ Eq (gauge EmptyCollection.emptyCollection) 0
  -/
  ext
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    x✝ : E
    ⊢ Eq (gauge EmptyCollection.emptyCollection x✝) (0 x✝)
  -/
  simp only [gauge_def', Real.sInf_empty, mem_empty_iff_false, Pi.zero_apply, sep_false]
  /-
    🎉 no goals
  -/


theorem gauge_of_subset_zero (h : s ⊆ 0) : gauge s = 0 := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    h : HasSubset.Subset s 0
    ⊢ Eq (gauge s) 0
  -/
  obtain rfl | rfl := subset_singleton_iff_eq.1 h
  /-
    case inl
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    h : HasSubset.Subset EmptyCollection.emptyCollection 0
    ⊢ Eq (gauge EmptyCollection.emptyCollection) 0
  -/
  exacts [gauge_empty, gauge_zero']
  /-
    🎉 no goals
  -/


/-- The gauge is always nonnegative. -/
theorem gauge_nonneg (x : E) : 0 ≤ gauge s x :=
  Real.sInf_nonneg fun _ hx => hx.1.le


theorem gauge_neg (symmetric : ∀ x ∈ s, -x ∈ s) (x : E) : gauge s (-x) = gauge s x := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    x : E
    ⊢ Eq (gauge s (Neg.neg x)) (gauge s x)
  -/
  have : ∀ x, -x ∈ s ↔ x ∈ s := fun x => ⟨fun h => by simpa using symmetric _ h, symmetric x⟩
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    x : E
    this : ∀ (x : E), Iff (Membership.mem s (Neg.neg x)) (Membership.mem s x)
    ⊢ Eq (gauge s (Neg.neg x)) (gauge s x)
  -/
  simp_rw [gauge_def', smul_neg, this]
  /-
    🎉 no goals
  -/


theorem gauge_neg_set_neg (x : E) : gauge (-s) (-x) = gauge s x := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    ⊢ Eq (gauge (Neg.neg s) (Neg.neg x)) (gauge s x)
  -/
  simp_rw [gauge_def', smul_neg, neg_mem_neg]
  /-
    🎉 no goals
  -/


theorem gauge_neg_set_eq_gauge_neg (x : E) : gauge (-s) x = gauge s (-x) := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    ⊢ Eq (gauge (Neg.neg s) x) (gauge s (Neg.neg x))
  -/
  rw [← gauge_neg_set_neg, neg_neg]
  /-
    🎉 no goals
  -/


theorem gauge_le_of_mem (ha : 0 ≤ a) (hx : x ∈ a • s) : gauge s x ≤ a := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    ha : LE.le 0 a
    hx : Membership.mem (HSMul.hSMul a s) x
    ⊢ LE.le (gauge s x) a
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      x : E
      ha : LE.le 0 0
      hx : Membership.mem (HSMul.hSMul 0 s) x
      ⊢ LE.le (gauge s x) 0
    -/
  · rw [mem_singleton_iff.1 (zero_smul_set_subset _ hx), gauge_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      x : E
      a : Real
      ha : LE.le 0 a
      hx : Membership.mem (HSMul.hSMul a s) x
      ha' : LT.lt 0 a
      ⊢ LE.le (gauge s x) a
    -/
  · exact csInf_le gauge_set_bddBelow ⟨ha', hx⟩
    /-
      🎉 no goals
    -/


theorem gauge_le_eq (hs₁ : Convex ℝ s) (hs₀ : (0 : E) ∈ s) (hs₂ : Absorbent ℝ s) (ha : 0 ≤ a) :
    { x | gauge s x ≤ a } = ⋂ (r : ℝ) (_ : a < r), r • s := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : Absorbent Real s
    ha : LE.le 0 a
    ⊢ Eq (setOf fun x => LE.le (gauge s x) a) (Set.iInter fun r => Set.iInter fun  …
  -/
  ext x
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : Absorbent Real s
    ha : LE.le 0 a
    x : E
    ⊢ Iff (Membership.mem (setOf fun x => LE.le (gauge s x) a) x) (Membership.mem  …
  -/
  simp_rw [Set.mem_iInter, Set.mem_setOf_eq]
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : Absorbent Real s
    ha : LE.le 0 a
    x : E
    ⊢ Iff (LE.le (gauge s x) a) (∀ (i : Real), LT.lt a i → Membership.mem (HSMul.h …
  -/
  refine ⟨fun h r hr => ?_, fun h => le_of_forall_pos_lt_add fun ε hε => ?_⟩
    /-
      case h.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      ⊢ Membership.mem (HSMul.hSMul r s) x
    -/
  · have hr' := ha.trans_lt hr
    /-
      case h.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      ⊢ Membership.mem (HSMul.hSMul r s) x
    -/
    rw [mem_smul_set_iff_inv_smul_mem₀ hr'.ne']
    /-
      case h.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      ⊢ Membership.mem s (HSMul.hSMul (Inv.inv r) x)
    -/
    obtain ⟨δ, δ_pos, hδr, hδ⟩ := exists_lt_of_gauge_lt hs₂ (h.trans_lt hr)
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      δ : Real
      δ_pos : LT.lt 0 δ
      hδr : LT.lt δ r
      hδ : Membership.mem (HSMul.hSMul δ s) x
      ⊢ Membership.mem s (HSMul.hSMul (Inv.inv r) x)
    -/
    suffices (r⁻¹ * δ) • δ⁻¹ • x ∈ s by rwa [smul_smul, mul_inv_cancel_right₀ δ_pos.ne'] at this
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      δ : Real
      δ_pos : LT.lt 0 δ
      hδr : LT.lt δ r
      hδ : Membership.mem (HSMul.hSMul δ s) x
      ⊢ Membership.mem s (HSMul.hSMul (HMul.hMul (Inv.inv r) δ) (HSMul.hSMul (Inv.in …
    -/
    rw [mem_smul_set_iff_inv_smul_mem₀ δ_pos.ne'] at hδ
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      δ : Real
      δ_pos : LT.lt 0 δ
      hδr : LT.lt δ r
      hδ : Membership.mem s (HSMul.hSMul (Inv.inv δ) x)
      ⊢ Membership.mem s (HSMul.hSMul (HMul.hMul (Inv.inv r) δ) (HSMul.hSMul (Inv.in …
    -/
    refine hs₁.smul_mem_of_zero_mem hs₀ hδ ⟨by positivity, ?_⟩
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      δ : Real
      δ_pos : LT.lt 0 δ
      hδr : LT.lt δ r
      hδ : Membership.mem s (HSMul.hSMul (Inv.inv δ) x)
      ⊢ LE.le (HMul.hMul (Inv.inv r) δ) 1
    -/
    rw [inv_mul_le_iff₀ hr', mul_one]
    /-
      case h.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : LE.le (gauge s x) a
      r : Real
      hr : LT.lt a r
      hr' : LT.lt 0 r
      δ : Real
      δ_pos : LT.lt 0 δ
      hδr : LT.lt δ r
      hδ : Membership.mem s (HSMul.hSMul (Inv.inv δ) x)
      ⊢ LE.le δ r
    -/
    exact hδr.le
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₁ : Convex Real s
      hs₀ : Membership.mem s 0
      hs₂ : Absorbent Real s
      ha : LE.le 0 a
      x : E
      h : ∀ (i : Real), LT.lt a i → Membership.mem (HSMul.hSMul i s) x
      ε : Real
      hε : LT.lt 0 ε
      ⊢ LT.lt (gauge s x) (HAdd.hAdd a ε)
    -/
  · have hε' := (lt_add_iff_pos_right a).2 (half_pos hε)
    exact
      (gauge_le_of_mem (ha.trans hε'.le) <| h _ hε').trans_lt (add_lt_add_left (half_lt_self hε) _)


theorem gauge_lt_eq' (absorbs : Absorbent ℝ s) (a : ℝ) :
    { x | gauge s x < a } = ⋃ (r : ℝ) (_ : 0 < r) (_ : r < a), r • s := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    a : Real
    ⊢ Eq (setOf fun x => LT.lt (gauge s x) a) (Set.iUnion fun r => Set.iUnion fun  …
  -/
  ext
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    a : Real
    x✝ : E
    ⊢ Iff (Membership.mem (setOf fun x => LT.lt (gauge s x) a) x✝) (Membership.mem …
  -/
  simp_rw [mem_setOf, mem_iUnion, exists_prop]
  exact
    ⟨exists_lt_of_gauge_lt absorbs, fun ⟨r, hr₀, hr₁, hx⟩ =>
      (gauge_le_of_mem hr₀.le hx).trans_lt hr₁⟩


theorem gauge_lt_eq (absorbs : Absorbent ℝ s) (a : ℝ) :
    { x | gauge s x < a } = ⋃ r ∈ Set.Ioo 0 (a : ℝ), r • s := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    a : Real
    ⊢ Eq (setOf fun x => LT.lt (gauge s x) a) (Set.iUnion fun r => Set.iUnion fun  …
  -/
  ext
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    a : Real
    x✝ : E
    ⊢ Iff (Membership.mem (setOf fun x => LT.lt (gauge s x) a) x✝) (Membership.mem …
  -/
  simp_rw [mem_setOf, mem_iUnion, exists_prop, mem_Ioo, and_assoc]
  exact
    ⟨exists_lt_of_gauge_lt absorbs, fun ⟨r, hr₀, hr₁, hx⟩ =>
      (gauge_le_of_mem hr₀.le hx).trans_lt hr₁⟩


theorem mem_openSegment_of_gauge_lt_one (absorbs : Absorbent ℝ s) (hgauge : gauge s x < 1) :
    ∃ y ∈ s, x ∈ openSegment ℝ 0 y := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    absorbs : Absorbent Real s
    hgauge : LT.lt (gauge s x) 1
    ⊢ Exists fun y => And (Membership.mem s y) (Membership.mem (openSegment Real 0 …
  -/
  rcases exists_lt_of_gauge_lt absorbs hgauge with ⟨r, hr₀, hr₁, y, hy, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    y : E
    hy : Membership.mem s y
    hgauge : LT.lt (gauge s ((fun x => HSMul.hSMul r x) y)) 1
    ⊢ Exists fun y_1 => And (Membership.mem s y_1) (Membership.mem (openSegment Re …
  -/
  refine ⟨y, hy, 1 - r, r, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    absorbs : Absorbent Real s
    r : Real
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    y : E
    hy : Membership.mem s y
    hgauge : LT.lt (gauge s ((fun x => HSMul.hSMul r x) y)) 1
    ⊢ And (LT.lt 0 (HSub.hSub 1 r)) (And (LT.lt 0 r) (And (Eq (HAdd.hAdd (HSub.hSu …
  -/
  simp [*]
  /-
    🎉 no goals
  -/


theorem gauge_lt_one_subset_self (hs : Convex ℝ s) (h₀ : (0 : E) ∈ s) (absorbs : Absorbent ℝ s) :
    { x | gauge s x < 1 } ⊆ s := fun _x hx ↦
  let ⟨_y, hys, hx⟩ := mem_openSegment_of_gauge_lt_one absorbs hx
  hs.openSegment_subset h₀ hys hx


theorem gauge_le_one_of_mem {x : E} (hx : x ∈ s) : gauge s x ≤ 1 :=
                                    /-
                                      E : Type u_2
                                      inst✝¹ : AddCommGroup E
                                      inst✝ : Module Real E
                                      s : Set E
                                      x : E
                                      hx : Membership.mem s x
                                      ⊢ Membership.mem (HSMul.hSMul 1 s) x
                                    -/
  gauge_le_of_mem zero_le_one <| by rwa [one_smul]
                                    /-
                                      🎉 no goals
                                    -/


/-- Gauge is subadditive. -/
theorem gauge_add_le (hs : Convex ℝ s) (absorbs : Absorbent ℝ s) (x y : E) :
    gauge s (x + y) ≤ gauge s x + gauge s y := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    hs : Convex Real s
    absorbs : Absorbent Real s
    x y : E
    ⊢ LE.le (gauge s (HAdd.hAdd x y)) (HAdd.hAdd (gauge s x) (gauge s y))
  -/
  refine le_of_forall_pos_lt_add fun ε hε => ?_
  obtain ⟨a, ha, ha', x, hx, rfl⟩ :=
    exists_lt_of_gauge_lt absorbs (lt_add_of_pos_right (gauge s x) (half_pos hε))
  obtain ⟨b, hb, hb', y, hy, rfl⟩ :=
    exists_lt_of_gauge_lt absorbs (lt_add_of_pos_right (gauge s y) (half_pos hε))
  calc
    gauge s (a • x + b • y) ≤ a + b := gauge_le_of_mem (by positivity) <| by
      rw [hs.add_smul ha.le hb.le]
      exact add_mem_add (smul_mem_smul_set hx) (smul_mem_smul_set hy)
    _ < gauge s (a • x) + gauge s (b • y) + ε := by linarith


theorem self_subset_gauge_le_one : s ⊆ { x | gauge s x ≤ 1 } := fun _ => gauge_le_one_of_mem


theorem Convex.gauge_le (hs : Convex ℝ s) (h₀ : (0 : E) ∈ s) (absorbs : Absorbent ℝ s) (a : ℝ) :
    Convex ℝ { x | gauge s x ≤ a } := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    hs : Convex Real s
    h₀ : Membership.mem s 0
    absorbs : Absorbent Real s
    a : Real
    ⊢ Convex Real (setOf fun x => LE.le (gauge s x) a)
  -/
  by_cases ha : 0 ≤ a
    /-
      case pos
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      hs : Convex Real s
      h₀ : Membership.mem s 0
      absorbs : Absorbent Real s
      a : Real
      ha : LE.le 0 a
      ⊢ Convex Real (setOf fun x => LE.le (gauge s x) a)
    -/
  · rw [gauge_le_eq hs h₀ absorbs ha]
    /-
      case pos
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      hs : Convex Real s
      h₀ : Membership.mem s 0
      absorbs : Absorbent Real s
      a : Real
      ha : LE.le 0 a
      ⊢ Convex Real (Set.iInter fun r => Set.iInter fun x => HSMul.hSMul r s)
    -/
    exact convex_iInter fun i => convex_iInter fun _ => hs.smul _
    /-
      🎉 no goals
    -/
  · -- Porting note: `convert` needed help
    /-
      case neg
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      hs : Convex Real s
      h₀ : Membership.mem s 0
      absorbs : Absorbent Real s
      a : Real
      ha : Not (LE.le 0 a)
      ⊢ Convex Real (setOf fun x => LE.le (gauge s x) a)
    -/
    convert convex_empty (𝕜 := ℝ) (E := E)
    /-
      case h.e'_6
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      hs : Convex Real s
      h₀ : Membership.mem s 0
      absorbs : Absorbent Real s
      a : Real
      ha : Not (LE.le 0 a)
      ⊢ Eq (setOf fun x => LE.le (gauge s x) a) EmptyCollection.emptyCollection
    -/
    exact eq_empty_iff_forall_not_mem.2 fun x hx => ha <| (gauge_nonneg _).trans hx
    /-
      🎉 no goals
    -/


theorem Balanced.starConvex (hs : Balanced ℝ s) : StarConvex ℝ 0 s :=
  starConvex_zero_iff.2 fun _ hx a ha₀ ha₁ =>
             /-
               E : Type u_2
               inst✝¹ : AddCommGroup E
               inst✝ : Module Real E
               s : Set E
               hs : Balanced Real s
               x✝ : E
               hx : Membership.mem s x✝
               a : Real
               ha₀ : LE.le 0 a
               ha₁ : LE.le a 1
               ⊢ LE.le (Norm.norm a) 1
             -/
    hs _ (by rwa [Real.norm_of_nonneg ha₀]) (smul_mem_smul_set hx)
             /-
               🎉 no goals
             -/


theorem le_gauge_of_not_mem (hs₀ : StarConvex ℝ 0 s) (hs₂ : Absorbs ℝ s {x}) (hx : x ∉ a • s) :
    a ≤ gauge s x := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    hs₀ : StarConvex Real 0 s
    hs₂ : Absorbs Real s (Singleton.singleton x)
    hx : Not (Membership.mem (HSMul.hSMul a s) x)
    ⊢ LE.le a (gauge s x)
  -/
  rw [starConvex_zero_iff] at hs₀
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    hs₂ : Absorbs Real s (Singleton.singleton x)
    hx : Not (Membership.mem (HSMul.hSMul a s) x)
    ⊢ LE.le a (gauge s x)
  -/
  obtain ⟨r, hr, h⟩ := hs₂.exists_pos
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    hs₂ : Absorbs Real s (Singleton.singleton x)
    hx : Not (Membership.mem (HSMul.hSMul a s) x)
    r : Real
    hr : GT.gt r 0
    h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
    ⊢ LE.le a (gauge s x)
  -/
  refine le_csInf ⟨r, hr, singleton_subset_iff.1 <| h _ (Real.norm_of_nonneg hr.le).ge⟩ ?_
  /-
    case intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    x : E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    hs₂ : Absorbs Real s (Singleton.singleton x)
    hx : Not (Membership.mem (HSMul.hSMul a s) x)
    r : Real
    hr : GT.gt r 0
    h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
    ⊢ ∀ (b : Real), Membership.mem (setOf fun r => And (LT.lt 0 r) (Membership.mem …
  -/
  rintro b ⟨hb, x, hx', rfl⟩
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    r : Real
    hr : GT.gt r 0
    b : Real
    hb : LT.lt 0 b
    x : E
    hx' : Membership.mem s x
    hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
    hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
    h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
    ⊢ LE.le a b
  -/
  refine not_lt.1 fun hba => hx ?_
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    r : Real
    hr : GT.gt r 0
    b : Real
    hb : LT.lt 0 b
    x : E
    hx' : Membership.mem s x
    hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
    hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
    h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
    hba : LT.lt b a
    ⊢ Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x)
  -/
  have ha := hb.trans hba
  /-
    case intro.intro.intro.intro.intro
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s : Set E
    a : Real
    hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
    r : Real
    hr : GT.gt r 0
    b : Real
    hb : LT.lt 0 b
    x : E
    hx' : Membership.mem s x
    hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
    hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
    h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
    hba : LT.lt b a
    ha : LT.lt 0 a
    ⊢ Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x)
  -/
  refine ⟨(a⁻¹ * b) • x, hs₀ hx' (by positivity) ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
      r : Real
      hr : GT.gt r 0
      b : Real
      hb : LT.lt 0 b
      x : E
      hx' : Membership.mem s x
      hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
      hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
      h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
      hba : LT.lt b a
      ha : LT.lt 0 a
      ⊢ LE.le (HMul.hMul (Inv.inv a) b) 1
    -/
  · rw [← div_eq_inv_mul]
    /-
      case intro.intro.intro.intro.intro.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
      r : Real
      hr : GT.gt r 0
      b : Real
      hb : LT.lt 0 b
      x : E
      hx' : Membership.mem s x
      hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
      hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
      h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
      hba : LT.lt b a
      ha : LT.lt 0 a
      ⊢ LE.le (HDiv.hDiv b a) 1
    -/
    exact div_le_one_of_le₀ hba.le ha.le
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
      r : Real
      hr : GT.gt r 0
      b : Real
      hb : LT.lt 0 b
      x : E
      hx' : Membership.mem s x
      hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
      hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
      h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
      hba : LT.lt b a
      ha : LT.lt 0 a
      ⊢ Eq ((fun x => HSMul.hSMul a x) (HSMul.hSMul (HMul.hMul (Inv.inv a) b) x)) (( …
    -/
  · dsimp only
    /-
      case intro.intro.intro.intro.intro.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s : Set E
      a : Real
      hs₀ : ∀ ⦃x : E⦄, Membership.mem s x → ∀ ⦃a : Real⦄, LE.le 0 a → LE.le a 1 → Me …
      r : Real
      hr : GT.gt r 0
      b : Real
      hb : LT.lt 0 b
      x : E
      hx' : Membership.mem s x
      hs₂ : Absorbs Real s (Singleton.singleton ((fun x => HSMul.hSMul b x) x))
      hx : Not (Membership.mem (HSMul.hSMul a s) ((fun x => HSMul.hSMul b x) x))
      h : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset (Singleton.singleto …
      hba : LT.lt b a
      ha : LT.lt 0 a
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul (HMul.hMul (Inv.inv a) b) x)) (HSMul.hSMul b x)
    -/
    rw [← mul_smul, mul_inv_cancel_left₀ ha.ne']
    /-
      🎉 no goals
    -/


theorem one_le_gauge_of_not_mem (hs₁ : StarConvex ℝ 0 s) (hs₂ : Absorbs ℝ s {x}) (hx : x ∉ s) :
    1 ≤ gauge s x :=
                                    /-
                                      E : Type u_2
                                      inst✝¹ : AddCommGroup E
                                      inst✝ : Module Real E
                                      s : Set E
                                      x : E
                                      hs₁ : StarConvex Real 0 s
                                      hs₂ : Absorbs Real s (Singleton.singleton x)
                                      hx : Not (Membership.mem s x)
                                      ⊢ Not (Membership.mem (HSMul.hSMul 1 s) x)
                                    -/
  le_gauge_of_not_mem hs₁ hs₂ <| by rwa [one_smul]
                                    /-
                                      🎉 no goals
                                    -/


theorem gauge_smul_of_nonneg [MulActionWithZero α E] [IsScalarTower α ℝ (Set E)] {s : Set E} {a : α}
    (ha : 0 ≤ a) (x : E) : gauge s (a • x) = a • gauge s x := by
  /-
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ⊢ Eq (gauge s (HSMul.hSMul a x)) (HSMul.hSMul a (gauge s x))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      x : E
      ha : LE.le 0 0
      ⊢ Eq (gauge s (HSMul.hSMul 0 x)) (HSMul.hSMul 0 (gauge s x))
    -/
  · rw [zero_smul, gauge_zero, zero_smul]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ha' : LT.lt 0 a
    ⊢ Eq (gauge s (HSMul.hSMul a x)) (HSMul.hSMul a (gauge s x))
  -/
  rw [gauge_def', gauge_def', ← Real.sInf_smul_of_nonneg ha]
  /-
    case inr
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ha' : LT.lt 0 a
    ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
  -/
  congr 1
  /-
    case inr.e_a
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ha' : LT.lt 0 a
    ⊢ Eq (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membership.mem s (HSM …
  -/
  ext r
  /-
    case inr.e_a.h
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ha' : LT.lt 0 a
    r : Real
    ⊢ Iff (Membership.mem (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Memb …
  -/
  simp_rw [Set.mem_smul_set, Set.mem_sep_iff]
  /-
    case inr.e_a.h
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    α : Type u_3
    inst✝⁴ : LinearOrderedField α
    inst✝³ : MulActionWithZero α Real
    inst✝² : OrderedSMul α Real
    inst✝¹ : MulActionWithZero α E
    inst✝ : IsScalarTower α Real (Set E)
    s : Set E
    a : α
    ha : LE.le 0 a
    x : E
    ha' : LT.lt 0 a
    r : Real
    ⊢ Iff (And (Membership.mem (Set.Ioi 0) r) (Membership.mem s (HSMul.hSMul (Inv. …
  -/
  constructor
    /-
      case inr.e_a.h.mp
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      ⊢ And (Membership.mem (Set.Ioi 0) r) (Membership.mem s (HSMul.hSMul (Inv.inv r …
    -/
  · rintro ⟨hr, hx⟩
    /-
      case inr.e_a.h.mp.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) (HSMul.hSMul a x))
      ⊢ Exists fun y => And (And (Membership.mem (Set.Ioi 0) y) (Membership.mem s (H …
    -/
    simp_rw [mem_Ioi] at hr ⊢
    /-
      case inr.e_a.h.mp.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) (HSMul.hSMul a x))
      hr : LT.lt 0 r
      ⊢ Exists fun y => And (And (LT.lt 0 y) (Membership.mem s (HSMul.hSMul (Inv.inv …
    -/
    rw [← mem_smul_set_iff_inv_smul_mem₀ hr.ne'] at hx
    /-
      case inr.e_a.h.mp.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hx : Membership.mem (HSMul.hSMul r s) (HSMul.hSMul a x)
      hr : LT.lt 0 r
      ⊢ Exists fun y => And (And (LT.lt 0 y) (Membership.mem s (HSMul.hSMul (Inv.inv …
    -/
    have := smul_pos (inv_pos.2 ha') hr
    /-
      case inr.e_a.h.mp.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hx : Membership.mem (HSMul.hSMul r s) (HSMul.hSMul a x)
      hr : LT.lt 0 r
      this : LT.lt 0 (HSMul.hSMul (Inv.inv a) r)
      ⊢ Exists fun y => And (And (LT.lt 0 y) (Membership.mem s (HSMul.hSMul (Inv.inv …
    -/
    refine ⟨a⁻¹ • r, ⟨this, ?_⟩, smul_inv_smul₀ ha'.ne' _⟩
    rwa [← mem_smul_set_iff_inv_smul_mem₀ this.ne', smul_assoc,
      mem_smul_set_iff_inv_smul_mem₀ (inv_ne_zero ha'.ne'), inv_inv]
    /-
      case inr.e_a.h.mpr
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      ⊢ (Exists fun y => And (And (Membership.mem (Set.Ioi 0) y) (Membership.mem s ( …
    -/
  · rintro ⟨r, ⟨hr, hx⟩, rfl⟩
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) x)
      ⊢ And (Membership.mem (Set.Ioi 0) (HSMul.hSMul a r)) (Membership.mem s (HSMul. …
    -/
    rw [mem_Ioi] at hr ⊢
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) x)
      ⊢ And (LT.lt 0 (HSMul.hSMul a r)) (Membership.mem s (HSMul.hSMul (Inv.inv (HSM …
    -/
    rw [← mem_smul_set_iff_inv_smul_mem₀ hr.ne'] at hx
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem (HSMul.hSMul r s) x
      ⊢ And (LT.lt 0 (HSMul.hSMul a r)) (Membership.mem s (HSMul.hSMul (Inv.inv (HSM …
    -/
    have := smul_pos ha' hr
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem (HSMul.hSMul r s) x
      this : LT.lt 0 (HSMul.hSMul a r)
      ⊢ And (LT.lt 0 (HSMul.hSMul a r)) (Membership.mem s (HSMul.hSMul (Inv.inv (HSM …
    -/
    refine ⟨this, ?_⟩
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem (HSMul.hSMul r s) x
      this : LT.lt 0 (HSMul.hSMul a r)
      ⊢ Membership.mem s (HSMul.hSMul (Inv.inv (HSMul.hSMul a r)) (HSMul.hSMul a x))
    -/
    rw [← mem_smul_set_iff_inv_smul_mem₀ this.ne', smul_assoc]
    /-
      case inr.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module Real E
      α : Type u_3
      inst✝⁴ : LinearOrderedField α
      inst✝³ : MulActionWithZero α Real
      inst✝² : OrderedSMul α Real
      inst✝¹ : MulActionWithZero α E
      inst✝ : IsScalarTower α Real (Set E)
      s : Set E
      a : α
      ha : LE.le 0 a
      x : E
      ha' : LT.lt 0 a
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem (HSMul.hSMul r s) x
      this : LT.lt 0 (HSMul.hSMul a r)
      ⊢ Membership.mem (HSMul.hSMul a (HSMul.hSMul r s)) (HSMul.hSMul a x)
    -/
    exact smul_mem_smul_set hx
    /-
      🎉 no goals
    -/


theorem gauge_smul_left_of_nonneg [MulActionWithZero α E] [SMulCommClass α ℝ ℝ]
    [IsScalarTower α ℝ ℝ] [IsScalarTower α ℝ E] {s : Set E} {a : α} (ha : 0 ≤ a) :
    gauge (a • s) = a⁻¹ • gauge s := by
  /-
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ⊢ Eq (gauge (HSMul.hSMul a s)) (HSMul.hSMul (Inv.inv a) (gauge s))
  -/
  obtain rfl | ha' := ha.eq_or_lt
    /-
      case inl
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      ha : LE.le 0 0
      ⊢ Eq (gauge (HSMul.hSMul 0 s)) (HSMul.hSMul (Inv.inv 0) (gauge s))
    -/
  · rw [inv_zero, zero_smul, gauge_of_subset_zero (zero_smul_set_subset _)]
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    ⊢ Eq (gauge (HSMul.hSMul a s)) (HSMul.hSMul (Inv.inv a) (gauge s))
  -/
  ext x
  /-
    case inr.h
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    x : E
    ⊢ Eq (gauge (HSMul.hSMul a s) x) (HSMul.hSMul (Inv.inv a) (gauge s) x)
  -/
  rw [gauge_def', Pi.smul_apply, gauge_def', ← Real.sInf_smul_of_nonneg (inv_nonneg.2 ha)]
  /-
    case inr.h
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    x : E
    ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membersh …
  -/
  congr 1
  /-
    case inr.h.e_a
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    x : E
    ⊢ Eq (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Membership.mem (HSMul …
  -/
  ext r
  /-
    case inr.h.e_a.h
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    x : E
    r : Real
    ⊢ Iff (Membership.mem (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Memb …
  -/
  simp_rw [Set.mem_smul_set, Set.mem_sep_iff]
  /-
    case inr.h.e_a.h
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : MulActionWithZero α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    a : α
    ha : LE.le 0 a
    ha' : LT.lt 0 a
    x : E
    r : Real
    ⊢ Iff (And (Membership.mem (Set.Ioi 0) r) (Exists fun y => And (Membership.mem …
  -/
  constructor
    /-
      case inr.h.e_a.h.mp
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      ⊢ And (Membership.mem (Set.Ioi 0) r) (Exists fun y => And (Membership.mem s y) …
    -/
  · rintro ⟨hr, y, hy, h⟩
    /-
      case inr.h.e_a.h.mp.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      y : E
      hy : Membership.mem s y
      h : Eq (HSMul.hSMul a y) (HSMul.hSMul (Inv.inv r) x)
      ⊢ Exists fun y => And (And (Membership.mem (Set.Ioi 0) y) (Membership.mem s (H …
    -/
    simp_rw [mem_Ioi] at hr ⊢
    /-
      case inr.h.e_a.h.mp.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      y : E
      hy : Membership.mem s y
      h : Eq (HSMul.hSMul a y) (HSMul.hSMul (Inv.inv r) x)
      hr : LT.lt 0 r
      ⊢ Exists fun y => And (And (LT.lt 0 y) (Membership.mem s (HSMul.hSMul (Inv.inv …
    -/
    refine ⟨a • r, ⟨smul_pos ha' hr, ?_⟩, inv_smul_smul₀ ha'.ne' _⟩
    /-
      case inr.h.e_a.h.mp.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      y : E
      hy : Membership.mem s y
      h : Eq (HSMul.hSMul a y) (HSMul.hSMul (Inv.inv r) x)
      hr : LT.lt 0 r
      ⊢ Membership.mem s (HSMul.hSMul (Inv.inv (HSMul.hSMul a r)) x)
    -/
    rwa [smul_inv₀, smul_assoc, ← h, inv_smul_smul₀ ha'.ne']
    /-
      🎉 no goals
    -/
    /-
      case inr.h.e_a.h.mpr
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      ⊢ (Exists fun y => And (And (Membership.mem (Set.Ioi 0) y) (Membership.mem s ( …
    -/
  · rintro ⟨r, ⟨hr, hx⟩, rfl⟩
    /-
      case inr.h.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      hr : Membership.mem (Set.Ioi 0) r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) x)
      ⊢ And (Membership.mem (Set.Ioi 0) (HSMul.hSMul (Inv.inv a) r)) (Exists fun y = …
    -/
    rw [mem_Ioi] at hr ⊢
    /-
      case inr.h.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) x)
      ⊢ And (LT.lt 0 (HSMul.hSMul (Inv.inv a) r)) (Exists fun y => And (Membership.m …
    -/
    refine ⟨smul_pos (inv_pos.2 ha') hr, r⁻¹ • x, hx, ?_⟩
    /-
      case inr.h.e_a.h.mpr.intro.intro.intro
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : MulActionWithZero α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      a : α
      ha : LE.le 0 a
      ha' : LT.lt 0 a
      x : E
      r : Real
      hr : LT.lt 0 r
      hx : Membership.mem s (HSMul.hSMul (Inv.inv r) x)
      ⊢ Eq (HSMul.hSMul a (HSMul.hSMul (Inv.inv r) x)) (HSMul.hSMul (Inv.inv (HSMul. …
    -/
    rw [smul_inv₀, smul_assoc, inv_inv]
    /-
      🎉 no goals
    -/


theorem gauge_smul_left [Module α E] [SMulCommClass α ℝ ℝ] [IsScalarTower α ℝ ℝ]
    [IsScalarTower α ℝ E] {s : Set E} (symmetric : ∀ x ∈ s, -x ∈ s) (a : α) :
    gauge (a • s) = |a|⁻¹ • gauge s := by
  /-
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : Module α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    a : α
    ⊢ Eq (gauge (HSMul.hSMul a s)) (HSMul.hSMul (Inv.inv (abs a)) (gauge s))
  -/
  rw [← gauge_smul_left_of_nonneg (abs_nonneg a)]
  /-
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module Real E
    α : Type u_3
    inst✝⁶ : LinearOrderedField α
    inst✝⁵ : MulActionWithZero α Real
    inst✝⁴ : OrderedSMul α Real
    inst✝³ : Module α E
    inst✝² : SMulCommClass α Real Real
    inst✝¹ : IsScalarTower α Real Real
    inst✝ : IsScalarTower α Real E
    s : Set E
    symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
    a : α
    ⊢ Eq (gauge (HSMul.hSMul a s)) (gauge (HSMul.hSMul (abs a) s))
  -/
  obtain h | h := abs_choice a
    /-
      case inl
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) a
      ⊢ Eq (gauge (HSMul.hSMul a s)) (gauge (HSMul.hSMul (abs a) s))
    -/
  · rw [h]
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      ⊢ Eq (gauge (HSMul.hSMul a s)) (gauge (HSMul.hSMul (abs a) s))
    -/
  · rw [h, Set.neg_smul_set, ← Set.smul_set_neg]
    -- Porting note: was congr
    /-
      case inr
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      ⊢ Eq (gauge (HSMul.hSMul a s)) (gauge (HSMul.hSMul a (Neg.neg s)))
    -/
    apply congr_arg
    /-
      case inr.h
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      ⊢ Eq (HSMul.hSMul a s) (HSMul.hSMul a (Neg.neg s))
    -/
    apply congr_arg
    /-
      case inr.h.h
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      ⊢ Eq s (Neg.neg s)
    -/
    ext y
    /-
      case inr.h.h.h
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      y : E
      ⊢ Iff (Membership.mem s y) (Membership.mem (Neg.neg s) y)
    -/
    refine ⟨symmetric _, fun hy => ?_⟩
    /-
      case inr.h.h.h
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      y : E
      hy : Membership.mem (Neg.neg s) y
      ⊢ Membership.mem s y
    -/
    rw [← neg_neg y]
    /-
      case inr.h.h.h
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module Real E
      α : Type u_3
      inst✝⁶ : LinearOrderedField α
      inst✝⁵ : MulActionWithZero α Real
      inst✝⁴ : OrderedSMul α Real
      inst✝³ : Module α E
      inst✝² : SMulCommClass α Real Real
      inst✝¹ : IsScalarTower α Real Real
      inst✝ : IsScalarTower α Real E
      s : Set E
      symmetric : ∀ (x : E), Membership.mem s x → Membership.mem s (Neg.neg x)
      a : α
      h : Eq (abs a) (Neg.neg a)
      y : E
      hy : Membership.mem (Neg.neg s) y
      ⊢ Membership.mem s (Neg.neg (Neg.neg y))
    -/
    exact symmetric _ hy
    /-
      🎉 no goals
    -/


theorem gauge_norm_smul (hs : Balanced 𝕜 s) (r : 𝕜) (x : E) :
    gauge s (‖r‖ • x) = gauge s (r • x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    ⊢ Eq (gauge s (HSMul.hSMul (Norm.norm r) x)) (gauge s (HSMul.hSMul r x))
  -/
  unfold gauge
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    ⊢ Eq (InfSet.sInf (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSMul.h …
  -/
  congr with θ
  /-
    case e_a.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    θ : Real
    ⊢ Iff (Membership.mem (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSM …
  -/
  rw [@RCLike.real_smul_eq_coe_smul 𝕜]
  /-
    case e_a.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    θ : Real
    ⊢ Iff (Membership.mem (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSM …
  -/
  refine and_congr_right fun hθ => (hs.smul _).smul_mem_iff ?_
  /-
    case e_a.h
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    θ : Real
    hθ : LT.lt 0 θ
    ⊢ Eq (Norm.norm ↑(Norm.norm r)) (Norm.norm r)
  -/
  rw [RCLike.norm_ofReal, abs_norm]
  /-
    🎉 no goals
  -/


/-- If `s` is balanced, then the Minkowski functional is ℂ-homogeneous. -/
theorem gauge_smul (hs : Balanced 𝕜 s) (r : 𝕜) (x : E) : gauge s (r • x) = ‖r‖ * gauge s x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : RCLike 𝕜
    inst✝¹ : Module 𝕜 E
    inst✝ : IsScalarTower Real 𝕜 E
    hs : Balanced 𝕜 s
    r : 𝕜
    x : E
    ⊢ Eq (gauge s (HSMul.hSMul r x)) (HMul.hMul (Norm.norm r) (gauge s x))
  -/
  rw [← smul_eq_mul, ← gauge_smul_of_nonneg (norm_nonneg r), gauge_norm_smul hs]
  /-
    🎉 no goals
  -/


theorem comap_gauge_nhds_zero_le (ha : Absorbent ℝ s) (hb : Bornology.IsVonNBounded ℝ s) :
    comap (gauge s) (𝓝 0) ≤ 𝓝 0 := fun u hu ↦ by
  /-
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    ⊢ Membership.mem (Filter.comap (gauge s) (nhds 0)) u
  -/
  rcases (hb hu).exists_pos with ⟨r, hr₀, hr⟩
  /-
    case intro.intro
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    r : Real
    hr₀ : GT.gt r 0
    hr : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c u)
    ⊢ Membership.mem (Filter.comap (gauge s) (nhds 0)) u
  -/
  filter_upwards [preimage_mem_comap (gt_mem_nhds (inv_pos.2 hr₀))] with x (hx : gauge s x < r⁻¹)
  /-
    case h
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    r : Real
    hr₀ : GT.gt r 0
    hr : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c u)
    x : E
    hx : LT.lt (gauge s x) (Inv.inv r)
    ⊢ Membership.mem u x
  -/
  rcases exists_lt_of_gauge_lt ha hx with ⟨c, hc₀, hcr, y, hy, rfl⟩
  /-
    case h.intro.intro.intro.intro.intro
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    r : Real
    hr₀ : GT.gt r 0
    hr : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c u)
    c : Real
    hc₀ : LT.lt 0 c
    hcr : LT.lt c (Inv.inv r)
    y : E
    hy : Membership.mem s y
    hx : LT.lt (gauge s ((fun x => HSMul.hSMul c x) y)) (Inv.inv r)
    ⊢ Membership.mem u ((fun x => HSMul.hSMul c x) y)
  -/
  have hrc := (lt_inv_comm₀ hr₀ hc₀).2 hcr
  /-
    case h.intro.intro.intro.intro.intro
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    r : Real
    hr₀ : GT.gt r 0
    hr : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c u)
    c : Real
    hc₀ : LT.lt 0 c
    hcr : LT.lt c (Inv.inv r)
    y : E
    hy : Membership.mem s y
    hx : LT.lt (gauge s ((fun x => HSMul.hSMul c x) y)) (Inv.inv r)
    hrc : LT.lt r (Inv.inv c)
    ⊢ Membership.mem u ((fun x => HSMul.hSMul c x) y)
  -/
  rcases hr c⁻¹ (hrc.le.trans (le_abs_self _)) hy with ⟨z, hz, rfl⟩
  /-
    case h.intro.intro.intro.intro.intro.intro.intro
    E : Type u_2
    inst✝² : AddCommGroup E
    inst✝¹ : Module Real E
    s : Set E
    inst✝ : TopologicalSpace E
    ha : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    u : Set E
    hu : Membership.mem (nhds 0) u
    r : Real
    hr₀ : GT.gt r 0
    hr : ∀ (c : Real), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c u)
    c : Real
    hc₀ : LT.lt 0 c
    hcr : LT.lt c (Inv.inv r)
    hrc : LT.lt r (Inv.inv c)
    z : E
    hz : Membership.mem u z
    hy : Membership.mem s ((fun x => HSMul.hSMul (Inv.inv c) x) z)
    hx : LT.lt (gauge s ((fun x => HSMul.hSMul c x) ((fun x => HSMul.hSMul (Inv.in …
    ⊢ Membership.mem u ((fun x => HSMul.hSMul c x) ((fun x => HSMul.hSMul (Inv.inv …
  -/
  simpa only [smul_inv_smul₀ hc₀.ne']
  /-
    🎉 no goals
  -/


theorem gauge_eq_zero (hs : Absorbent ℝ s) (hb : Bornology.IsVonNBounded ℝ s) :
    gauge s x = 0 ↔ x = 0 := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    x : E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    hs : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    ⊢ Iff (Eq (gauge s x) 0) (Eq x 0)
  -/
  refine ⟨fun h₀ ↦ by_contra fun (hne : x ≠ 0) ↦ ?_, fun h ↦ h.symm ▸ gauge_zero⟩
  have : {x}ᶜ ∈ comap (gauge s) (𝓝 0) :=
    comap_gauge_nhds_zero_le hs hb (isOpen_compl_singleton.mem_nhds hne.symm)
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    x : E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    hs : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    h₀ : Eq (gauge s x) 0
    hne : Ne x 0
    this : Membership.mem (Filter.comap (gauge s) (nhds 0)) (HasCompl.compl (Singl …
    ⊢ False
  -/
  rcases ((nhds_basis_zero_abs_sub_lt _).comap _).mem_iff.1 this with ⟨r, hr₀, hr⟩
  /-
    case intro.intro
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    x : E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    hs : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    h₀ : Eq (gauge s x) 0
    hne : Ne x 0
    this : Membership.mem (Filter.comap (gauge s) (nhds 0)) (HasCompl.compl (Singl …
    r : Real
    hr₀ : LT.lt 0 r
    hr : HasSubset.Subset (Set.preimage (gauge s) (setOf fun b => LT.lt (abs b) r) …
    ⊢ False
  -/
  exact hr (by simpa [h₀]) rfl
  /-
    🎉 no goals
  -/


theorem gauge_pos (hs : Absorbent ℝ s) (hb : Bornology.IsVonNBounded ℝ s) :
    0 < gauge s x ↔ x ≠ 0 := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    x : E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    hs : Absorbent Real s
    hb : Bornology.IsVonNBounded Real s
    ⊢ Iff (LT.lt 0 (gauge s x)) (Ne x 0)
  -/
  simp only [(gauge_nonneg _).gt_iff_ne, Ne, gauge_eq_zero hs hb]
  /-
    🎉 no goals
  -/


open Filter in
theorem interior_subset_gauge_lt_one (s : Set E) : interior s ⊆ { x | gauge s x < 1 } := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    s : Set E
    ⊢ HasSubset.Subset (interior s) (setOf fun x => LT.lt (gauge s x) 1)
  -/
  intro x hx
  have H₁ : Tendsto (fun r : ℝ ↦ r⁻¹ • x) (𝓝[<] 1) (𝓝 ((1 : ℝ)⁻¹ • x)) :=
    ((tendsto_id.inv₀ one_ne_zero).smul tendsto_const_nhds).mono_left inf_le_left
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    s : Set E
    x : E
    hx : Membership.mem (interior s) x
    H₁ : Filter.Tendsto (fun r => HSMul.hSMul (Inv.inv r) x) (nhdsWithin 1 (Set.Ii …
    ⊢ Membership.mem (setOf fun x => LT.lt (gauge s x) 1) x
  -/
  rw [inv_one, one_smul] at H₁
  have H₂ : ∀ᶠ r in 𝓝[<] (1 : ℝ), x ∈ r • s ∧ 0 < r ∧ r < 1 := by
    filter_upwards [H₁ (mem_interior_iff_mem_nhds.1 hx), Ioo_mem_nhdsLT one_pos] with r h₁ h₂
    exact ⟨(mem_smul_set_iff_inv_smul_mem₀ h₂.1.ne' _ _).2 h₁, h₂⟩
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    s : Set E
    x : E
    hx : Membership.mem (interior s) x
    H₁ : Filter.Tendsto (fun r => HSMul.hSMul (Inv.inv r) x) (nhdsWithin 1 (Set.Ii …
    H₂ : Filter.Eventually (fun r => And (Membership.mem (HSMul.hSMul r s) x) (And …
    ⊢ Membership.mem (setOf fun x => LT.lt (gauge s x) 1) x
  -/
  rcases H₂.exists with ⟨r, hxr, hr₀, hr₁⟩
  /-
    case intro.intro.intro
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    s : Set E
    x : E
    hx : Membership.mem (interior s) x
    H₁ : Filter.Tendsto (fun r => HSMul.hSMul (Inv.inv r) x) (nhdsWithin 1 (Set.Ii …
    H₂ : Filter.Eventually (fun r => And (Membership.mem (HSMul.hSMul r s) x) (And …
    r : Real
    hxr : Membership.mem (HSMul.hSMul r s) x
    hr₀ : LT.lt 0 r
    hr₁ : LT.lt r 1
    ⊢ Membership.mem (setOf fun x => LT.lt (gauge s x) 1) x
  -/
  exact (gauge_le_of_mem hr₀.le hxr).trans_lt hr₁
  /-
    🎉 no goals
  -/


theorem gauge_lt_one_eq_self_of_isOpen (hs₁ : Convex ℝ s) (hs₀ : (0 : E) ∈ s) (hs₂ : IsOpen s) :
    { x | gauge s x < 1 } = s := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : IsOpen s
    ⊢ Eq (setOf fun x => LT.lt (gauge s x) 1) s
  -/
  refine (gauge_lt_one_subset_self hs₁ ‹_› <| absorbent_nhds_zero <| hs₂.mem_nhds hs₀).antisymm ?_
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : IsOpen s
    ⊢ HasSubset.Subset s (setOf fun x => LT.lt (gauge s x) 1)
  -/
  convert interior_subset_gauge_lt_one s
  /-
    case h.e'_3
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs₁ : Convex Real s
    hs₀ : Membership.mem s 0
    hs₂ : IsOpen s
    ⊢ Eq s (interior s)
  -/
  exact hs₂.interior_eq.symm
  /-
    🎉 no goals
  -/


theorem gauge_lt_one_of_mem_of_isOpen (hs₂ : IsOpen s) {x : E} (hx : x ∈ s) :
    gauge s x < 1 :=
                                       /-
                                         E : Type u_2
                                         inst✝³ : AddCommGroup E
                                         inst✝² : Module Real E
                                         s : Set E
                                         inst✝¹ : TopologicalSpace E
                                         inst✝ : ContinuousSMul Real E
                                         hs₂ : IsOpen s
                                         x : E
                                         hx : Membership.mem s x
                                         ⊢ Membership.mem (interior s) x
                                       -/
  interior_subset_gauge_lt_one s <| by rwa [hs₂.interior_eq]
                                       /-
                                         🎉 no goals
                                       -/


theorem gauge_lt_of_mem_smul (x : E) (ε : ℝ) (hε : 0 < ε) (hs₂ : IsOpen s) (hx : x ∈ ε • s) :
    gauge s x < ε := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    x : E
    ε : Real
    hε : LT.lt 0 ε
    hs₂ : IsOpen s
    hx : Membership.mem (HSMul.hSMul ε s) x
    ⊢ LT.lt (gauge s x) ε
  -/
  have : ε⁻¹ • x ∈ s := by rwa [← mem_smul_set_iff_inv_smul_mem₀ hε.ne']
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    x : E
    ε : Real
    hε : LT.lt 0 ε
    hs₂ : IsOpen s
    hx : Membership.mem (HSMul.hSMul ε s) x
    this : Membership.mem s (HSMul.hSMul (Inv.inv ε) x)
    ⊢ LT.lt (gauge s x) ε
  -/
  have h_gauge_lt := gauge_lt_one_of_mem_of_isOpen hs₂ this
  rwa [gauge_smul_of_nonneg (inv_nonneg.2 hε.le), smul_eq_mul, inv_mul_lt_iff₀ hε, mul_one]
    at h_gauge_lt


theorem mem_closure_of_gauge_le_one (hc : Convex ℝ s) (hs₀ : 0 ∈ s) (ha : Absorbent ℝ s)
    (h : gauge s x ≤ 1) : x ∈ closure s := by
  have : ∀ᶠ r : ℝ in 𝓝[<] 1, r • x ∈ s := by
    filter_upwards [Ico_mem_nhdsLT one_pos] with r ⟨hr₀, hr₁⟩
    apply gauge_lt_one_subset_self hc hs₀ ha
    rw [mem_setOf_eq, gauge_smul_of_nonneg hr₀]
    exact mul_lt_one_of_nonneg_of_lt_one_left hr₀ hr₁ h
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    x : E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem s 0
    ha : Absorbent Real s
    h : LE.le (gauge s x) 1
    this : Filter.Eventually (fun r => Membership.mem s (HSMul.hSMul r x)) (nhdsWi …
    ⊢ Membership.mem (closure s) x
  -/
  refine mem_closure_of_tendsto ?_ this
  exact Filter.Tendsto.mono_left (Continuous.tendsto' (by fun_prop) _ _ (one_smul _ _))
    inf_le_left


theorem mem_frontier_of_gauge_eq_one (hc : Convex ℝ s) (hs₀ : 0 ∈ s) (ha : Absorbent ℝ s)
    (h : gauge s x = 1) : x ∈ frontier s :=
  ⟨mem_closure_of_gauge_le_one hc hs₀ ha h.le, fun h' ↦
    (interior_subset_gauge_lt_one s h').out.ne h⟩


theorem tendsto_gauge_nhds_zero' (hs : s ∈ 𝓝 0) : Tendsto (gauge s) (𝓝 0) (𝓝[≥] 0) := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : Membership.mem (nhds 0) s
    ⊢ Filter.Tendsto (gauge s) (nhds 0) (nhdsWithin 0 (Set.Ici 0))
  -/
  refine nhdsGE_basis_Icc.tendsto_right_iff.2 fun ε hε ↦ ?_
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : Membership.mem (nhds 0) s
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.Icc 0 ε) (gauge s x)) (nhds 0)
  -/
  rw [← set_smul_mem_nhds_zero_iff hε.ne'] at hs
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    ε : Real
    hs : Membership.mem (nhds 0) (HSMul.hSMul ε s)
    hε : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => Membership.mem (Set.Icc 0 ε) (gauge s x)) (nhds 0)
  -/
  filter_upwards [hs] with x hx
  /-
    case h
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    ε : Real
    hs : Membership.mem (nhds 0) (HSMul.hSMul ε s)
    hε : LT.lt 0 ε
    x : E
    hx : Membership.mem (HSMul.hSMul ε s) x
    ⊢ Membership.mem (Set.Icc 0 ε) (gauge s x)
  -/
  exact ⟨gauge_nonneg _, gauge_le_of_mem hε.le hx⟩
  /-
    🎉 no goals
  -/


theorem tendsto_gauge_nhds_zero (hs : s ∈ 𝓝 0) : Tendsto (gauge s) (𝓝 0) (𝓝 0) :=
  (tendsto_gauge_nhds_zero' hs).mono_right inf_le_left


/-- If `s` is a neighborhood of the origin, then `gauge s` is continuous at the origin.
See also `continuousAt_gauge`. -/
theorem continuousAt_gauge_zero (hs : s ∈ 𝓝 0) : ContinuousAt (gauge s) 0 := by
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : Membership.mem (nhds 0) s
    ⊢ ContinuousAt (gauge s) 0
  -/
  rw [ContinuousAt, gauge_zero]
  /-
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    s : Set E
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : Membership.mem (nhds 0) s
    ⊢ Filter.Tendsto (gauge s) (nhds 0) (nhds 0)
  -/
  exact tendsto_gauge_nhds_zero hs
  /-
    🎉 no goals
  -/


theorem comap_gauge_nhds_zero (hb : Bornology.IsVonNBounded ℝ s) (h₀ : s ∈ 𝓝 0) :
    comap (gauge s) (𝓝 0) = 𝓝 0 :=
  (comap_gauge_nhds_zero_le (absorbent_nhds_zero h₀) hb).antisymm
    (tendsto_gauge_nhds_zero h₀).le_comap


/-- If `s` is a convex neighborhood of the origin in a topological real vector space, then `gauge s`
is continuous. If the ambient space is a normed space, then `gauge s` is Lipschitz continuous, see
`Convex.lipschitz_gauge`. -/
theorem continuousAt_gauge (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) : ContinuousAt (gauge s) x := by
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ⊢ ContinuousAt (gauge s) x
  -/
  have ha : Absorbent ℝ s := absorbent_nhds_zero hs₀
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ha : Absorbent Real s
    ⊢ ContinuousAt (gauge s) x
  -/
  refine (nhds_basis_Icc_pos _).tendsto_right_iff.2 fun ε hε₀ ↦ ?_
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ha : Absorbent Real s
    ε : Real
    hε₀ : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Icc (HSub.hSub (gauge s x) …
  -/
  rw [← map_add_left_nhds_zero, eventually_map]
  have : ε • s ∩ -(ε • s) ∈ 𝓝 0 :=
    inter_mem ((set_smul_mem_nhds_zero_iff hε₀.ne').2 hs₀)
      (neg_mem_nhds_zero _ ((set_smul_mem_nhds_zero_iff hε₀.ne').2 hs₀))
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ha : Absorbent Real s
    ε : Real
    hε₀ : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Inter.inter (HSMul.hSMul ε s) (Neg.neg (HSMul. …
    ⊢ Filter.Eventually (fun a => Membership.mem (Set.Icc (HSub.hSub (gauge s x) ε …
  -/
  filter_upwards [this] with y hy
  /-
    case h
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ha : Absorbent Real s
    ε : Real
    hε₀ : LT.lt 0 ε
    this : Membership.mem (nhds 0) (Inter.inter (HSMul.hSMul ε s) (Neg.neg (HSMul. …
    y : E
    hy : Membership.mem (Inter.inter (HSMul.hSMul ε s) (Neg.neg (HSMul.hSMul ε s)) …
    ⊢ Membership.mem (Set.Icc (HSub.hSub (gauge s x) ε) (HAdd.hAdd (gauge s x) ε)) …
  -/
  constructor
    /-
      case h.left
      E : Type u_2
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module Real E
      s : Set E
      x : E
      inst✝² : TopologicalSpace E
      inst✝¹ : TopologicalAddGroup E
      inst✝ : ContinuousSMul Real E
      hc : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      ha : Absorbent Real s
      ε : Real
      hε₀ : LT.lt 0 ε
      this : Membership.mem (nhds 0) (Inter.inter (HSMul.hSMul ε s) (Neg.neg (HSMul. …
      y : E
      hy : Membership.mem (Inter.inter (HSMul.hSMul ε s) (Neg.neg (HSMul.hSMul ε s)) …
      ⊢ LE.le (HSub.hSub (gauge s x) ε) (gauge s (HAdd.hAdd x y))
    -/
  · rw [sub_le_iff_le_add]
    calc
      gauge s x = gauge s (x + y + (-y)) := by simp
      _ ≤ gauge s (x + y) + gauge s (-y) := gauge_add_le hc ha _ _
      _ ≤ gauge s (x + y) + ε := add_le_add_left (gauge_le_of_mem hε₀.le (mem_neg.1 hy.2)) _
  · calc
      gauge s (x + y) ≤ gauge s x + gauge s y := gauge_add_le hc ha _ _
      _ ≤ gauge s x + ε := add_le_add_left (gauge_le_of_mem hε₀.le hy.1) _


/-- If `s` is a convex neighborhood of the origin in a topological real vector space, then `gauge s`
is continuous. If the ambient space is a normed space, then `gauge s` is Lipschitz continuous, see
`Convex.lipschitz_gauge`. -/
@[continuity]
theorem continuous_gauge (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) : Continuous (gauge s) :=
  continuous_iff_continuousAt.2 fun _ ↦ continuousAt_gauge hc hs₀


theorem gauge_lt_one_eq_interior (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) :
    { x | gauge s x < 1 } = interior s := by
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ⊢ Eq (setOf fun x => LT.lt (gauge s x) 1) (interior s)
  -/
  refine Subset.antisymm (fun x hx ↦ ?_) (interior_subset_gauge_lt_one s)
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    x : E
    hx : Membership.mem (setOf fun x => LT.lt (gauge s x) 1) x
    ⊢ Membership.mem (interior s) x
  -/
  rcases mem_openSegment_of_gauge_lt_one (absorbent_nhds_zero hs₀) hx with ⟨y, hys, hxy⟩
  /-
    case intro.intro
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    x : E
    hx : Membership.mem (setOf fun x => LT.lt (gauge s x) 1) x
    y : E
    hys : Membership.mem s y
    hxy : Membership.mem (openSegment Real 0 y) x
    ⊢ Membership.mem (interior s) x
  -/
  exact hc.openSegment_interior_self_subset_interior (mem_interior_iff_mem_nhds.2 hs₀) hys hxy
  /-
    🎉 no goals
  -/


theorem gauge_lt_one_iff_mem_interior (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) :
    gauge s x < 1 ↔ x ∈ interior s :=
  Set.ext_iff.1 (gauge_lt_one_eq_interior hc hs₀) _


theorem gauge_le_one_iff_mem_closure (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) :
    gauge s x ≤ 1 ↔ x ∈ closure s :=
  ⟨mem_closure_of_gauge_le_one hc (mem_of_mem_nhds hs₀) (absorbent_nhds_zero hs₀), fun h ↦
    le_on_closure (fun _ ↦ gauge_le_one_of_mem) (continuous_gauge hc hs₀).continuousOn
      continuousOn_const h⟩


theorem gauge_eq_one_iff_mem_frontier (hc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) :
    gauge s x = 1 ↔ x ∈ frontier s := by
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ⊢ Iff (Eq (gauge s x) 1) (Membership.mem (frontier s) x)
  -/
  rw [eq_iff_le_not_lt, gauge_le_one_iff_mem_closure hc hs₀, gauge_lt_one_iff_mem_interior hc hs₀]
  /-
    E : Type u_2
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    s : Set E
    x : E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    hc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ⊢ Iff (And (Membership.mem (closure s) x) (Not (Membership.mem (interior s) x) …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- `gauge s` as a seminorm when `s` is balanced, convex and absorbent. -/
@[simps!]
def gaugeSeminorm (hs₀ : Balanced 𝕜 s) (hs₁ : Convex ℝ s) (hs₂ : Absorbent ℝ s) : Seminorm 𝕜 E :=
  Seminorm.of (gauge s) (gauge_add_le hs₁ hs₂) (gauge_smul hs₀)


theorem gaugeSeminorm_lt_one_of_isOpen (hs : IsOpen s) {x : E} (hx : x ∈ s) :
    gaugeSeminorm hs₀ hs₁ hs₂ x < 1 :=
  gauge_lt_one_of_mem_of_isOpen hs hx


theorem gaugeSeminorm_ball_one (hs : IsOpen s) : (gaugeSeminorm hs₀ hs₁ hs₂).ball 0 1 = s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    hs₀ : Balanced 𝕜 s
    hs₁ : Convex Real s
    hs₂ : Absorbent Real s
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : IsOpen s
    ⊢ Eq ((gaugeSeminorm hs₀ hs₁ hs₂).ball 0 1) s
  -/
  rw [Seminorm.ball_zero_eq]
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module Real E
    s : Set E
    inst✝⁴ : RCLike 𝕜
    inst✝³ : Module 𝕜 E
    inst✝² : IsScalarTower Real 𝕜 E
    hs₀ : Balanced 𝕜 s
    hs₁ : Convex Real s
    hs₂ : Absorbent Real s
    inst✝¹ : TopologicalSpace E
    inst✝ : ContinuousSMul Real E
    hs : IsOpen s
    ⊢ Eq (setOf fun y => LT.lt ((gaugeSeminorm hs₀ hs₁ hs₂) y) 1) s
  -/
  exact gauge_lt_one_eq_self_of_isOpen hs₁ hs₂.zero_mem hs
  /-
    🎉 no goals
  -/


/-- Any seminorm arises as the gauge of its unit ball. -/
@[simp]
protected theorem Seminorm.gauge_ball (p : Seminorm ℝ E) : gauge (p.ball 0 1) = p := by
  /-
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    p : Seminorm Real E
    ⊢ Eq (gauge (p.ball 0 1)) ⇑p
  -/
  ext x
  /-
    case h
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    p : Seminorm Real E
    x : E
    ⊢ Eq (gauge (p.ball 0 1) x) (p x)
  -/
  obtain hp | hp := { r : ℝ | 0 < r ∧ x ∈ r • p.ball 0 1 }.eq_empty_or_nonempty
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      ⊢ Eq (gauge (p.ball 0 1) x) (p x)
    -/
  · rw [gauge, hp, Real.sInf_empty]
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      ⊢ Eq 0 (p x)
    -/
    by_contra h
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      h : Not (Eq 0 (p x))
      ⊢ False
    -/
    have hpx : 0 < p x := (apply_nonneg _ _).lt_of_ne h
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      h : Not (Eq 0 (p x))
      hpx : LT.lt 0 (p x)
      ⊢ False
    -/
    have hpx₂ : 0 < 2 * p x := mul_pos zero_lt_two hpx
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      h : Not (Eq 0 (p x))
      hpx : LT.lt 0 (p x)
      hpx₂ : LT.lt 0 (HMul.hMul 2 (p x))
      ⊢ False
    -/
    refine hp.subset ⟨hpx₂, (2 * p x)⁻¹ • x, ?_, smul_inv_smul₀ hpx₂.ne' _⟩
    rw [p.mem_ball_zero, map_smul_eq_mul, Real.norm_eq_abs, abs_of_pos (inv_pos.2 hpx₂),
      inv_mul_lt_iff₀ hpx₂, mul_one]
    /-
      case h.inl
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : Eq (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball …
      h : Not (Eq 0 (p x))
      hpx : LT.lt 0 (p x)
      hpx₂ : LT.lt 0 (HMul.hMul 2 (p x))
      ⊢ LT.lt (p x) (HMul.hMul 2 (p x))
    -/
    exact lt_mul_of_one_lt_left hpx one_lt_two
    /-
      🎉 no goals
    -/
  /-
    case h.inr
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    p : Seminorm Real E
    x : E
    hp : (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball 0  …
    ⊢ Eq (gauge (p.ball 0 1) x) (p x)
  -/
  refine IsGLB.csInf_eq ⟨fun r => ?_, fun r hr => le_of_forall_pos_le_add fun ε hε => ?_⟩ hp
    /-
      case h.inr.refine_1
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball 0  …
      r : Real
      ⊢ Membership.mem (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul  …
    -/
  · rintro ⟨hr, y, hy, rfl⟩
    /-
      case h.inr.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      r : Real
      hr : LT.lt 0 r
      y : E
      hy : Membership.mem (p.ball 0 1) y
      hp : (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSMul.hSMul r_1 (p.b …
      ⊢ LE.le (p ((fun x => HSMul.hSMul r x) y)) r
    -/
    rw [p.mem_ball_zero] at hy
    /-
      case h.inr.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      r : Real
      hr : LT.lt 0 r
      y : E
      hy : LT.lt (p y) 1
      hp : (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSMul.hSMul r_1 (p.b …
      ⊢ LE.le (p ((fun x => HSMul.hSMul r x) y)) r
    -/
    rw [map_smul_eq_mul, Real.norm_eq_abs, abs_of_pos hr]
    /-
      case h.inr.refine_1.intro.intro.intro
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      r : Real
      hr : LT.lt 0 r
      y : E
      hy : LT.lt (p y) 1
      hp : (setOf fun r_1 => And (LT.lt 0 r_1) (Membership.mem (HSMul.hSMul r_1 (p.b …
      ⊢ LE.le (HMul.hMul r (p y)) r
    -/
    exact mul_le_of_le_one_right hr.le hy.le
    /-
      🎉 no goals
    -/
  · have hpε : 0 < p x + ε :=
      -- Porting note: was `by positivity`
      add_pos_of_nonneg_of_pos (apply_nonneg _ _) hε
    /-
      case h.inr.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball 0  …
      r : Real
      hr : Membership.mem (lowerBounds (setOf fun r => And (LT.lt 0 r) (Membership.m …
      ε : Real
      hε : LT.lt 0 ε
      hpε : LT.lt 0 (HAdd.hAdd (p x) ε)
      ⊢ LE.le r (HAdd.hAdd (p x) ε)
    -/
    refine hr ⟨hpε, (p x + ε)⁻¹ • x, ?_, smul_inv_smul₀ hpε.ne' _⟩
    rw [p.mem_ball_zero, map_smul_eq_mul, Real.norm_eq_abs, abs_of_pos (inv_pos.2 hpε),
      inv_mul_lt_iff₀ hpε, mul_one]
    /-
      case h.inr.refine_2
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      p : Seminorm Real E
      x : E
      hp : (setOf fun r => And (LT.lt 0 r) (Membership.mem (HSMul.hSMul r (p.ball 0  …
      r : Real
      hr : Membership.mem (lowerBounds (setOf fun r => And (LT.lt 0 r) (Membership.m …
      ε : Real
      hε : LT.lt 0 ε
      hpε : LT.lt 0 (HAdd.hAdd (p x) ε)
      ⊢ LT.lt (p x) (HAdd.hAdd (p x) ε)
    -/
    exact lt_add_of_pos_right _ hε
    /-
      🎉 no goals
    -/


theorem Seminorm.gaugeSeminorm_ball (p : Seminorm ℝ E) :
    gaugeSeminorm (p.balanced_ball_zero 1) (p.convex_ball 0 1) (p.absorbent_ball_zero zero_lt_one) =
      p :=
  DFunLike.coe_injective p.gauge_ball


theorem gauge_unit_ball (x : E) : gauge (ball (0 : E) 1) x = ‖x‖ := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    ⊢ Eq (gauge (Metric.ball 0 1) x) (Norm.norm x)
  -/
  rw [← ball_normSeminorm ℝ, Seminorm.gauge_ball, coe_normSeminorm]
  /-
    🎉 no goals
  -/


theorem gauge_ball (hr : 0 ≤ r) (x : E) : gauge (ball (0 : E) r) x = ‖x‖ / r := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LE.le 0 r
    x : E
    ⊢ Eq (gauge (Metric.ball 0 r) x) (HDiv.hDiv (Norm.norm x) r)
  -/
  rcases hr.eq_or_lt with rfl | hr
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hr : LE.le 0 0
      ⊢ Eq (gauge (Metric.ball 0 0) x) (HDiv.hDiv (Norm.norm x) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
  · rw [← smul_unitBall_of_pos hr, gauge_smul_left, Pi.smul_apply, gauge_unit_ball, smul_eq_mul,
    abs_of_nonneg hr.le, div_eq_inv_mul]
    /-
      case inr.symmetric
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      r : Real
      hr✝ : LE.le 0 r
      x : E
      hr : LT.lt 0 r
      ⊢ ∀ (x : E), Membership.mem (Metric.ball 0 1) x → Membership.mem (Metric.ball  …
    -/
    simp_rw [mem_ball_zero_iff, norm_neg]
    /-
      case inr.symmetric
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      r : Real
      hr✝ : LE.le 0 r
      x : E
      hr : LT.lt 0 r
      ⊢ ∀ (x : E), LT.lt (Norm.norm x) 1 → LT.lt (Norm.norm x) 1
    -/
    exact fun _ => id
    /-
      🎉 no goals
    -/


@[simp]
theorem gauge_closure_zero : gauge (closure (0 : Set E)) = 0 := funext fun x ↦ by
  simp only [← singleton_zero, gauge_def', mem_closure_zero_iff_norm, norm_smul, mul_eq_zero,
    norm_eq_zero, inv_eq_zero]
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    x : E
    ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r …
  -/
  rcases (norm_nonneg x).eq_or_gt with hx | hx
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx : Eq (Norm.norm x) 0
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r …
    -/
  · convert csInf_Ioi (a := (0 : ℝ))
    /-
      case h.e'_2.h.e'_3
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx : Eq (Norm.norm x) 0
      ⊢ Eq (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r 0) (Eq (Norm …
    -/
    exact Set.ext fun r ↦ and_iff_left (.inr hx)
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx : LT.lt 0 (Norm.norm x)
      ⊢ Eq (InfSet.sInf (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r …
    -/
  · convert Real.sInf_empty
    /-
      case h.e'_2.h.e'_3
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hx : LT.lt 0 (Norm.norm x)
      ⊢ Eq (setOf fun r => And (Membership.mem (Set.Ioi 0) r) (Or (Eq r 0) (Eq (Norm …
    -/
    exact eq_empty_of_forall_not_mem fun r ⟨hr₀, hr⟩ ↦ hx.ne' <| hr.resolve_left hr₀.out.ne'
    /-
      🎉 no goals
    -/


@[simp]
theorem gauge_closedBall (hr : 0 ≤ r) (x : E) : gauge (closedBall (0 : E) r) x = ‖x‖ / r := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    r : Real
    hr : LE.le 0 r
    x : E
    ⊢ Eq (gauge (Metric.closedBall 0 r) x) (HDiv.hDiv (Norm.norm x) r)
  -/
  rcases hr.eq_or_lt with rfl | hr'
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      x : E
      hr : LE.le 0 0
      ⊢ Eq (gauge (Metric.closedBall 0 0) x) (HDiv.hDiv (Norm.norm x) 0)
    -/
  · rw [div_zero, closedBall_zero', singleton_zero, gauge_closure_zero]; rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    /-
      case inr
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      r : Real
      hr : LE.le 0 r
      x : E
      hr' : LT.lt 0 r
      ⊢ Eq (gauge (Metric.closedBall 0 r) x) (HDiv.hDiv (Norm.norm x) r)
    -/
  · apply le_antisymm
      /-
        case inr.a
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        ⊢ LE.le (gauge (Metric.closedBall 0 r) x) (HDiv.hDiv (Norm.norm x) r)
      -/
    · rw [← gauge_ball hr]
      /-
        case inr.a
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        ⊢ LE.le (gauge (Metric.closedBall 0 r) x) (gauge (Metric.ball 0 r) x)
      -/
      exact gauge_mono (absorbent_ball_zero hr') ball_subset_closedBall x
      /-
        🎉 no goals
      -/
    · suffices ∀ᶠ R in 𝓝[>] r, ‖x‖ / R ≤ gauge (closedBall 0 r) x by
        refine le_of_tendsto ?_ this
        exact tendsto_const_nhds.div inf_le_left hr'.ne'
      /-
        case inr.a
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        ⊢ Filter.Eventually (fun R => LE.le (HDiv.hDiv (Norm.norm x) R) (gauge (Metric …
      -/
      filter_upwards [self_mem_nhdsWithin] with R hR
      /-
        case h
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        R : Real
        hR : Membership.mem (Set.Ioi r) R
        ⊢ LE.le (HDiv.hDiv (Norm.norm x) R) (gauge (Metric.closedBall 0 r) x)
      -/
      rw [← gauge_ball (hr.trans hR.out.le)]
      /-
        case h
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        R : Real
        hR : Membership.mem (Set.Ioi r) R
        ⊢ LE.le (gauge (Metric.ball 0 R) x) (gauge (Metric.closedBall 0 r) x)
      -/
      refine gauge_mono ?_ (closedBall_subset_ball hR) _
      /-
        case h
        E : Type u_2
        inst✝¹ : SeminormedAddCommGroup E
        inst✝ : NormedSpace Real E
        r : Real
        hr : LE.le 0 r
        x : E
        hr' : LT.lt 0 r
        R : Real
        hR : Membership.mem (Set.Ioi r) R
        ⊢ Absorbent Real (Metric.closedBall 0 r)
      -/
      exact (absorbent_ball_zero hr').mono ball_subset_closedBall
      /-
        🎉 no goals
      -/


theorem mul_gauge_le_norm (hs : Metric.ball (0 : E) r ⊆ s) : r * gauge s x ≤ ‖x‖ := by
  /-
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    r : Real
    x : E
    hs : HasSubset.Subset (Metric.ball 0 r) s
    ⊢ LE.le (HMul.hMul r (gauge s x)) (Norm.norm x)
  -/
  obtain hr | hr := le_or_lt r 0
    /-
      case inl
      E : Type u_2
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace Real E
      s : Set E
      r : Real
      x : E
      hs : HasSubset.Subset (Metric.ball 0 r) s
      hr : LE.le r 0
      ⊢ LE.le (HMul.hMul r (gauge s x)) (Norm.norm x)
    -/
  · exact (mul_nonpos_of_nonpos_of_nonneg hr <| gauge_nonneg _).trans (norm_nonneg _)
    /-
      🎉 no goals
    -/
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    r : Real
    x : E
    hs : HasSubset.Subset (Metric.ball 0 r) s
    hr : LT.lt 0 r
    ⊢ LE.le (HMul.hMul r (gauge s x)) (Norm.norm x)
  -/
  rw [mul_comm, ← le_div_iff₀ hr, ← gauge_ball hr.le]
  /-
    case inr
    E : Type u_2
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    r : Real
    x : E
    hs : HasSubset.Subset (Metric.ball 0 r) s
    hr : LT.lt 0 r
    ⊢ LE.le (gauge s x) (gauge (Metric.ball 0 r) x)
  -/
  exact gauge_mono (absorbent_ball_zero hr) hs x
  /-
    🎉 no goals
  -/


theorem Convex.lipschitzWith_gauge {r : ℝ≥0} (hc : Convex ℝ s) (hr : 0 < r)
    (hs : Metric.ball (0 : E) r ⊆ s) : LipschitzWith r⁻¹ (gauge s) :=
  have : Absorbent ℝ (Metric.ball (0 : E) r) := absorbent_ball_zero hr
  LipschitzWith.of_le_add_mul _ fun x y =>
    calc
                                              /-
                                                E : Type u_2
                                                inst✝¹ : SeminormedAddCommGroup E
                                                inst✝ : NormedSpace Real E
                                                s : Set E
                                                r : NNReal
                                                hc : Convex Real s
                                                hr : LT.lt 0 r
                                                hs : HasSubset.Subset (Metric.ball 0 ↑r) s
                                                this : Absorbent Real (Metric.ball 0 ↑r)
                                                x y : E
                                                ⊢ Eq (gauge s x) (gauge s (HAdd.hAdd y (HSub.hSub x y)))
                                              -/
      gauge s x = gauge s (y + (x - y)) := by simp
                                              /-
                                                🎉 no goals
                                              -/
      _ ≤ gauge s y + gauge s (x - y) := gauge_add_le hc (this.mono hs) _ _
      _ ≤ gauge s y + ‖x - y‖ / r :=
        add_le_add_left ((gauge_mono this hs (x - y)).trans_eq (gauge_ball hr.le _)) _
                                           /-
                                             E : Type u_2
                                             inst✝¹ : SeminormedAddCommGroup E
                                             inst✝ : NormedSpace Real E
                                             s : Set E
                                             r : NNReal
                                             hc : Convex Real s
                                             hr : LT.lt 0 r
                                             hs : HasSubset.Subset (Metric.ball 0 ↑r) s
                                             this : Absorbent Real (Metric.ball 0 ↑r)
                                             x y : E
                                             ⊢ Eq (HAdd.hAdd (gauge s y) (HDiv.hDiv (Norm.norm (HSub.hSub x y)) ↑r)) (HAdd. …
                                           -/
      _ = gauge s y + r⁻¹ * dist x y := by rw [dist_eq_norm, div_eq_inv_mul, NNReal.coe_inv]
                                           /-
                                             🎉 no goals
                                           -/


theorem Convex.lipschitz_gauge (hc : Convex ℝ s) (h₀ : s ∈ 𝓝 (0 : E)) :
    ∃ K, LipschitzWith K (gauge s) :=
  let ⟨r, hr₀, hr⟩ := Metric.mem_nhds_iff.1 h₀
  ⟨(⟨r, hr₀.le⟩ : ℝ≥0)⁻¹, hc.lipschitzWith_gauge hr₀ hr⟩


theorem Convex.uniformContinuous_gauge (hc : Convex ℝ s) (h₀ : s ∈ 𝓝 (0 : E)) :
    UniformContinuous (gauge s) :=
  let ⟨_K, hK⟩ := hc.lipschitz_gauge h₀; hK.uniformContinuous


theorem le_gauge_of_subset_closedBall (hs : Absorbent ℝ s) (hr : 0 ≤ r) (hsr : s ⊆ closedBall 0 r) :
    ‖x‖ / r ≤ gauge s x := by
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    r : Real
    x : E
    hs : Absorbent Real s
    hr : LE.le 0 r
    hsr : HasSubset.Subset s (Metric.closedBall 0 r)
    ⊢ LE.le (HDiv.hDiv (Norm.norm x) r) (gauge s x)
  -/
  rw [← gauge_closedBall hr]
  /-
    E : Type u_2
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    s : Set E
    r : Real
    x : E
    hs : Absorbent Real s
    hr : LE.le 0 r
    hsr : HasSubset.Subset s (Metric.closedBall 0 r)
    ⊢ LE.le (gauge (Metric.closedBall 0 r) x) (gauge s x)
  -/
  exact gauge_mono hs hsr _
  /-
    🎉 no goals
  -/


