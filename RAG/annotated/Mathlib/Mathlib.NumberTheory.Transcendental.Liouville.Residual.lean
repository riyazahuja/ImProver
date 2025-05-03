theorem setOf_liouville_eq_iInter_iUnion :
    { x | Liouville x } =
      ⋂ n : ℕ, ⋃ (a : ℤ) (b : ℤ) (_ : 1 < b),
      ball ((a : ℝ) / b) (1 / (b : ℝ) ^ n) \ {(a : ℝ) / b} := by
  /-
    ⊢ Eq (setOf fun x => Liouville x) (Set.iInter fun n => Set.iUnion fun a => Set …
  -/
  ext x
  simp only [mem_iInter, mem_iUnion, Liouville, mem_setOf_eq, exists_prop, mem_diff,
    mem_singleton_iff, mem_ball, Real.dist_eq, and_comm]


theorem IsGδ.setOf_liouville : IsGδ { x | Liouville x } := by
  /-
    ⊢ IsGδ (setOf fun x => Liouville x)
  -/
  rw [setOf_liouville_eq_iInter_iUnion]
  /-
    ⊢ IsGδ (Set.iInter fun n => Set.iUnion fun a => Set.iUnion fun b => Set.iUnion …
  -/
  refine .iInter fun n => IsOpen.isGδ ?_
  /-
    n : Nat
    ⊢ IsOpen (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun x => SDiff.sd …
  -/
  refine isOpen_iUnion fun a => isOpen_iUnion fun b => isOpen_iUnion fun _hb => ?_
  /-
    n : Nat
    a b : Int
    _hb : LT.lt 1 b
    ⊢ IsOpen (SDiff.sdiff (Metric.ball (HDiv.hDiv ↑a ↑b) (HDiv.hDiv 1 (HPow.hPow ( …
  -/
  exact isOpen_ball.inter isClosed_singleton.isOpen_compl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-02-15")] alias isGδ_setOf_liouville := IsGδ.setOf_liouville


theorem setOf_liouville_eq_irrational_inter_iInter_iUnion :
    { x | Liouville x } =
      { x | Irrational x } ∩ ⋂ n : ℕ, ⋃ (a : ℤ) (b : ℤ) (_ : 1 < b),
      ball (a / b) (1 / (b : ℝ) ^ n) := by
  /-
    ⊢ Eq (setOf fun x => Liouville x) (Inter.inter (setOf fun x => Irrational x) ( …
  -/
  refine Subset.antisymm ?_ ?_
    /-
      case refine_1
      ⊢ HasSubset.Subset (setOf fun x => Liouville x) (Inter.inter (setOf fun x => I …
    -/
  · refine subset_inter (fun x hx => hx.irrational) ?_
    /-
      case refine_1
      ⊢ HasSubset.Subset (setOf fun x => Liouville x) (Set.iInter fun n => Set.iUnio …
    -/
    rw [setOf_liouville_eq_iInter_iUnion]
    /-
      case refine_1
      ⊢ HasSubset.Subset (Set.iInter fun n => Set.iUnion fun a => Set.iUnion fun b = …
    -/
    exact iInter_mono fun n => iUnion₂_mono fun a b => iUnion_mono fun _hb => diff_subset
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ⊢ HasSubset.Subset (Inter.inter (setOf fun x => Irrational x) (Set.iInter fun  …
    -/
  · simp only [inter_iInter, inter_iUnion, setOf_liouville_eq_iInter_iUnion]
    /-
      case refine_2
      ⊢ HasSubset.Subset (Set.iInter fun i => Set.iUnion fun i_1 => Set.iUnion fun i …
    -/
    refine iInter_mono fun n => iUnion₂_mono fun a b => iUnion_mono fun hb => ?_
    /-
      case refine_2
      n : Nat
      a b : Int
      hb : LT.lt 1 b
      ⊢ HasSubset.Subset (Inter.inter (setOf fun x => Irrational x) (Metric.ball (HD …
    -/
    rw [inter_comm]
    /-
      case refine_2
      n : Nat
      a b : Int
      hb : LT.lt 1 b
      ⊢ HasSubset.Subset (Inter.inter (Metric.ball (HDiv.hDiv ↑a ↑b) (HDiv.hDiv 1 (H …
    -/
    exact diff_subset_diff Subset.rfl (singleton_subset_iff.2 ⟨a / b, by norm_cast⟩)
    /-
      🎉 no goals
    -/


/-- The set of Liouville numbers is a residual set. -/
theorem eventually_residual_liouville : ∀ᶠ x in residual ℝ, Liouville x := by
  /-
    ⊢ Filter.Eventually (fun x => Liouville x) (residual Real)
  -/
  rw [Filter.Eventually, setOf_liouville_eq_irrational_inter_iInter_iUnion]
  /-
    ⊢ Membership.mem (residual Real) (Inter.inter (setOf fun x => Irrational x) (S …
  -/
  refine eventually_residual_irrational.and ?_
  /-
    ⊢ Filter.Eventually (Membership.mem (Set.iInter fun n => Set.iUnion fun a => S …
  -/
  refine residual_of_dense_Gδ ?_ (Rat.isDenseEmbedding_coe_real.dense.mono ?_)
  · exact .iInter fun n => IsOpen.isGδ <|
          isOpen_iUnion fun a => isOpen_iUnion fun b => isOpen_iUnion fun _hb => isOpen_ball
    /-
      case refine_2
      ⊢ HasSubset.Subset (Set.range Rat.cast) (setOf fun x => Membership.mem (Set.iI …
    -/
  · rintro _ ⟨r, rfl⟩
    /-
      case refine_2.intro
      r : Rat
      ⊢ Membership.mem (setOf fun x => Membership.mem (Set.iInter fun n => Set.iUnio …
    -/
    simp only [mem_iInter, mem_iUnion]
    /-
      case refine_2.intro
      r : Rat
      ⊢ Membership.mem (setOf fun x => ∀ (i : Nat), Exists fun i_1 => Exists fun i_2 …
    -/
    refine fun n => ⟨r.num * 2, r.den * 2, ?_, ?_⟩
      /-
        case refine_2.intro.refine_1
        r : Rat
        n : Nat
        ⊢ LT.lt 1 (HMul.hMul (↑r.den) 2)
      -/
    · have := r.pos; omega
                     /-
                       🎉 no goals
                     -/
      /-
        case refine_2.intro.refine_2
        r : Rat
        n : Nat
        ⊢ Membership.mem (Metric.ball (HDiv.hDiv ↑(HMul.hMul r.num 2) ↑(HMul.hMul (↑r. …
      -/
    · convert @mem_ball_self ℝ _ (r : ℝ) _ _
        /-
          case h.e'_4.h.e'_3
          r : Rat
          n : Nat
          ⊢ Eq (HDiv.hDiv ↑(HMul.hMul r.num 2) ↑(HMul.hMul (↑r.den) 2)) ↑r
        -/
      · push_cast
        -- Workaround for https://github.com/leanprover/lean4/pull/6438; this eliminates an
        -- `Expr.mdata` that would cause `norm_cast` to skip a numeral.
        /-
          case h.e'_4.h.e'_3
          r : Rat
          n : Nat
          ⊢ Eq (HDiv.hDiv (HMul.hMul (↑r.num) 2) (HMul.hMul (↑r.den) 2)) ↑r
        -/
        rw [Eq.refl (2 : ℝ)]
        /-
          case h.e'_4.h.e'_3
          r : Rat
          n : Nat
          ⊢ Eq (HDiv.hDiv (HMul.hMul (↑r.num) 2) (HMul.hMul (↑r.den) 2)) ↑r
        -/
        norm_cast
        /-
          case h.e'_4.h.e'_3
          r : Rat
          n : Nat
          ⊢ Eq (Rat.divInt (HMul.hMul r.num 2) ↑(HMul.hMul r.den 2)) r
        -/
        simp [Rat.divInt_mul_right (two_ne_zero), Rat.mkRat_self]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.intro.refine_2.convert_2
          r : Rat
          n : Nat
          ⊢ LT.lt 0 (HDiv.hDiv 1 (HPow.hPow (↑(HMul.hMul (↑r.den) 2)) n))
        -/
      · refine one_div_pos.2 (pow_pos (Int.cast_pos.2 ?_) _)
        /-
          case refine_2.intro.refine_2.convert_2
          r : Rat
          n : Nat
          ⊢ LT.lt 0 (HMul.hMul (↑r.den) 2)
        -/
        exact mul_pos (Int.natCast_pos.2 r.pos) zero_lt_two
        /-
          🎉 no goals
        -/


/-- The set of Liouville numbers in dense. -/
theorem dense_liouville : Dense { x | Liouville x } :=
  dense_of_mem_residual eventually_residual_liouville

