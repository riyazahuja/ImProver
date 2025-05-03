/-- The set of p-adic integers `ℤ_[p]` is totally bounded. -/
theorem totallyBounded_univ : TotallyBounded (Set.univ : Set ℤ_[p]) := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ TotallyBounded Set.univ
  -/
  refine Metric.totallyBounded_iff.mpr (fun ε hε ↦ ?_)
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ε : Real
    hε : GT.gt ε 0
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
  -/
  obtain ⟨k, hk⟩ := exists_pow_neg_lt p hε
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ε : Real
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
  -/
  refine ⟨Nat.cast '' Finset.range (p ^ k), Set.toFinite _, fun z _ ↦ ?_⟩
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ε : Real
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    z : PadicInt p
    x✝ : Membership.mem Set.univ z
    ⊢ Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y ε) z
  -/
  simp only [PadicInt, Set.mem_iUnion, Metric.mem_ball, exists_prop, Set.exists_mem_image]
  /-
    case intro
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ε : Real
    hε : GT.gt ε 0
    k : Nat
    hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
    z : PadicInt p
    x✝ : Membership.mem Set.univ z
    ⊢ Exists fun x => And (Membership.mem (↑(Finset.range (HPow.hPow p k))) x) (LT …
  -/
  refine ⟨z.appr k, ?_, ?_⟩
    /-
      case intro.refine_1
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      ε : Real
      hε : GT.gt ε 0
      k : Nat
      hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
      z : PadicInt p
      x✝ : Membership.mem Set.univ z
      ⊢ Membership.mem (↑(Finset.range (HPow.hPow p k))) (z.appr k)
    -/
  · simpa only [Finset.mem_coe, Finset.mem_range] using z.appr_lt k
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      p : Nat
      inst✝ : Fact (Nat.Prime p)
      ε : Real
      hε : GT.gt ε 0
      k : Nat
      hk : LT.lt (HPow.hPow (↑p) (Neg.neg ↑k)) ε
      z : PadicInt p
      x✝ : Membership.mem Set.univ z
      ⊢ LT.lt (Dist.dist z ↑(z.appr k)) ε
    -/
  · exact (((z - z.appr k).norm_le_pow_iff_mem_span_pow k).mpr (z.appr_spec k)).trans_lt hk
    /-
      🎉 no goals
    -/


/-- The set of p-adic integers `ℤ_[p]` is a compact topological space. -/
instance compactSpace : CompactSpace ℤ_[p] := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ CompactSpace (PadicInt p)
  -/
  rw [← isCompact_univ_iff, isCompact_iff_totallyBounded_isComplete]
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ And (TotallyBounded Set.univ) (IsComplete Set.univ)
  -/
  exact ⟨totallyBounded_univ p, complete_univ⟩
  /-
    🎉 no goals
  -/


/-- The field of p-adic numbers `ℚ_[p]` is a proper metric space. -/
instance : ProperSpace ℚ_[p] := by
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ ProperSpace (Padic p)
  -/
  suffices LocallyCompactSpace ℚ_[p] from .of_nontriviallyNormedField_of_weaklyLocallyCompactSpace _
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    ⊢ LocallyCompactSpace (Padic p)
  -/
  have : closedBall 0 1 ∈ 𝓝 (0 : ℚ_[p]) := closedBall_mem_nhds _ zero_lt_one
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Membership.mem (nhds 0) (Metric.closedBall 0 1)
    ⊢ LocallyCompactSpace (Padic p)
  -/
  simp only [closedBall, dist_eq_norm_sub, sub_zero] at this
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Membership.mem (nhds 0) (setOf fun y => LE.le (Norm.norm y) 1)
    ⊢ LocallyCompactSpace (Padic p)
  -/
  refine IsCompact.locallyCompactSpace_of_mem_nhds_of_addGroup ?_ this
  /-
    p : Nat
    inst✝ : Fact (Nat.Prime p)
    this : Membership.mem (nhds 0) (setOf fun y => LE.le (Norm.norm y) 1)
    ⊢ IsCompact (setOf fun y => LE.le (Norm.norm y) 1)
  -/
  simpa only [isCompact_iff_compactSpace] using PadicInt.compactSpace p
  /-
    🎉 no goals
  -/


