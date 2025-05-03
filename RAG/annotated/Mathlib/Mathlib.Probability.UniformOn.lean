/-- Given a set `s`, `uniformOn s` is the uniform measure on `s`, defined as the counting measure
conditioned by `s`. One should think of `uniformOn s t` as the proportion of `s` that is contained
in `t`.

This is a probability measure when `s` is finite and nonempty and is given by
`ProbabilityTheory.uniformOn_isProbabilityMeasure`. -/
def uniformOn (s : Set Ω) : Measure Ω :=
  Measure.count[|s]


@[deprecated (since := "2024-10-09")]
noncomputable alias condCount := uniformOn


instance {s : Set Ω} : IsZeroOrProbabilityMeasure (uniformOn s) := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    s✝ s : Set Ω
    ⊢ MeasureTheory.IsZeroOrProbabilityMeasure (ProbabilityTheory.uniformOn s)
  -/
  unfold uniformOn; infer_instance
                    /-
                      🎉 no goals
                    -/


@[simp]
                                                                   /-
                                                                     Ω : Type u_1
                                                                     inst✝ : MeasurableSpace Ω
                                                                     ⊢ Eq (ProbabilityTheory.uniformOn EmptyCollection.emptyCollection) 0
                                                                   -/
theorem uniformOn_empty_meas : (uniformOn ∅ : Measure Ω) = 0 := by simp [uniformOn]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-10-09")]
alias condCount_empty_meas := uniformOn_empty_meas


                                                              /-
                                                                Ω : Type u_1
                                                                inst✝ : MeasurableSpace Ω
                                                                s : Set Ω
                                                                ⊢ Eq ((ProbabilityTheory.uniformOn s) EmptyCollection.emptyCollection) 0
                                                              -/
theorem uniformOn_empty {s : Set Ω} : uniformOn s ∅ = 0 := by simp
                                                              /-
                                                                🎉 no goals
                                                              -/


@[deprecated (since := "2024-10-09")]
alias condCount_empty := uniformOn_empty


/-- See `uniformOn_eq_zero` for a version assuming `MeasurableSingletonClass Ω` instead of
`MeasurableSet s`. -/
@[simp] lemma uniformOn_eq_zero' (hs : MeasurableSet s) : uniformOn s = 0 ↔ s.Infinite ∨ s = ∅ := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Iff (Eq (ProbabilityTheory.uniformOn s) 0) (Or s.Infinite (Eq s EmptyCollect …
  -/
  simp [uniformOn, hs]
  /-
    🎉 no goals
  -/


/-- See `uniformOn_eq_zero'` for a version assuming `MeasurableSet s` instead of
`MeasurableSingletonClass Ω`. -/
@[simp] lemma uniformOn_eq_zero [MeasurableSingletonClass Ω] :
                                               /-
                                                 Ω : Type u_1
                                                 inst✝¹ : MeasurableSpace Ω
                                                 s : Set Ω
                                                 inst✝ : MeasurableSingletonClass Ω
                                                 ⊢ Iff (Eq (ProbabilityTheory.uniformOn s) 0) (Or s.Infinite (Eq s EmptyCollect …
                                               -/
    uniformOn s = 0 ↔ s.Infinite ∨ s = ∅ := by simp [uniformOn]
                                               /-
                                                 🎉 no goals
                                               -/


theorem finite_of_uniformOn_ne_zero {s t : Set Ω} (h : uniformOn s t ≠ 0) : s.Finite := by
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    s t : Set Ω
    h : Ne ((ProbabilityTheory.uniformOn s) t) 0
    ⊢ s.Finite
  -/
  by_contra hs'
  /-
    Ω : Type u_1
    inst✝ : MeasurableSpace Ω
    s t : Set Ω
    h : Ne ((ProbabilityTheory.uniformOn s) t) 0
    hs' : Not s.Finite
    ⊢ False
  -/
  simp [uniformOn, cond, Measure.count_apply_infinite hs'] at h
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias finite_of_condCount_ne_zero := finite_of_uniformOn_ne_zero


theorem uniformOn_univ [Fintype Ω] {s : Set Ω} :
    uniformOn Set.univ s = Measure.count s / Fintype.card Ω := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : Fintype Ω
    s : Set Ω
    ⊢ Eq ((ProbabilityTheory.uniformOn Set.univ) s) (HDiv.hDiv (MeasureTheory.Meas …
  -/
  rw [uniformOn, cond_apply MeasurableSet.univ, ← ENNReal.div_eq_inv_mul, Set.univ_inter]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : Fintype Ω
    s : Set Ω
    ⊢ Eq (HDiv.hDiv (MeasureTheory.Measure.count s) (MeasureTheory.Measure.count S …
  -/
  congr
  /-
    case e_a
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : Fintype Ω
    s : Set Ω
    ⊢ Eq (MeasureTheory.Measure.count Set.univ) ↑(Fintype.card Ω)
  -/
  rw [← Finset.coe_univ, Measure.count_apply, Finset.univ.tsum_subtype' fun _ => (1 : ENNReal)]
    /-
      case e_a
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : Fintype Ω
      s : Set Ω
      ⊢ Eq (Finset.univ.sum fun x => 1) ↑(Fintype.card Ω)
    -/
  · simp [Finset.card_univ]
    /-
      🎉 no goals
    -/
    /-
      case e_a
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : Fintype Ω
      s : Set Ω
      ⊢ MeasurableSet ↑Finset.univ
    -/
  · exact (@Finset.coe_univ Ω _).symm ▸ MeasurableSet.univ
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-09")]
alias condCount_univ := uniformOn_univ


theorem uniformOn_isProbabilityMeasure {s : Set Ω} (hs : s.Finite) (hs' : s.Nonempty) :
    IsProbabilityMeasure (uniformOn s) := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ⊢ MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.uniformOn s)
  -/
  apply cond_isProbabilityMeasure_of_finite
    /-
      case hcs
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s : Set Ω
      hs : s.Finite
      hs' : s.Nonempty
      ⊢ Ne (MeasureTheory.Measure.count s) 0
    -/
  · rwa [Measure.count_ne_zero_iff]
    /-
      🎉 no goals
    -/
    /-
      case hs
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s : Set Ω
      hs : s.Finite
      hs' : s.Nonempty
      ⊢ Ne (MeasureTheory.Measure.count s) Top.top
    -/
  · exact (Measure.count_apply_lt_top.2 hs).ne
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-09")]
alias condCount_isProbabilityMeasure := uniformOn_isProbabilityMeasure


theorem uniformOn_singleton (ω : Ω) (t : Set Ω) [Decidable (ω ∈ t)] :
    uniformOn {ω} t = if ω ∈ t then 1 else 0 := by
  rw [uniformOn, cond_apply (measurableSet_singleton ω), Measure.count_singleton, inv_one,
    one_mul]
  /-
    Ω : Type u_1
    inst✝² : MeasurableSpace Ω
    inst✝¹ : MeasurableSingletonClass Ω
    ω : Ω
    t : Set Ω
    inst✝ : Decidable (Membership.mem t ω)
    ⊢ Eq (MeasureTheory.Measure.count (Inter.inter (Singleton.singleton ω) t)) (it …
  -/
  split_ifs
    /-
      case pos
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : MeasurableSingletonClass Ω
      ω : Ω
      t : Set Ω
      inst✝ : Decidable (Membership.mem t ω)
      h✝ : Membership.mem t ω
      ⊢ Eq (MeasureTheory.Measure.count (Inter.inter (Singleton.singleton ω) t)) 1
    -/
  · rw [(by simpa : ({ω} : Set Ω) ∩ t = {ω}), Measure.count_singleton]
    /-
      🎉 no goals
    -/
    /-
      case neg
      Ω : Type u_1
      inst✝² : MeasurableSpace Ω
      inst✝¹ : MeasurableSingletonClass Ω
      ω : Ω
      t : Set Ω
      inst✝ : Decidable (Membership.mem t ω)
      h✝ : Not (Membership.mem t ω)
      ⊢ Eq (MeasureTheory.Measure.count (Inter.inter (Singleton.singleton ω) t)) 0
    -/
  · rw [(by simpa : ({ω} : Set Ω) ∩ t = ∅), Measure.count_empty]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-09")]
alias condCount_singleton := uniformOn_singleton


theorem uniformOn_inter_self (hs : s.Finite) : uniformOn s (s ∩ t) = uniformOn s t := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hs : s.Finite
    ⊢ Eq ((ProbabilityTheory.uniformOn s) (Inter.inter s t)) ((ProbabilityTheory.u …
  -/
  rw [uniformOn, cond_inter_self hs.measurableSet]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias condCount_inter_self := uniformOn_inter_self


theorem uniformOn_self (hs : s.Finite) (hs' : s.Nonempty) : uniformOn s s = 1 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ⊢ Eq ((ProbabilityTheory.uniformOn s) s) 1
  -/
  rw [uniformOn, cond_apply hs.measurableSet, Set.inter_self, ENNReal.inv_mul_cancel]
    /-
      case h0
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s : Set Ω
      hs : s.Finite
      hs' : s.Nonempty
      ⊢ Ne (MeasureTheory.Measure.count s) 0
    -/
  · rwa [Measure.count_ne_zero_iff]
    /-
      🎉 no goals
    -/
    /-
      case ht
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s : Set Ω
      hs : s.Finite
      hs' : s.Nonempty
      ⊢ Ne (MeasureTheory.Measure.count s) Top.top
    -/
  · exact (Measure.count_apply_lt_top.2 hs).ne
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-09")]
alias condCount_self := uniformOn_self


theorem uniformOn_eq_one_of (hs : s.Finite) (hs' : s.Nonempty) (ht : s ⊆ t) :
    uniformOn s t = 1 := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ht : HasSubset.Subset s t
    ⊢ Eq ((ProbabilityTheory.uniformOn s) t) 1
  -/
  haveI := uniformOn_isProbabilityMeasure hs hs'
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ht : HasSubset.Subset s t
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.uniformOn s)
    ⊢ Eq ((ProbabilityTheory.uniformOn s) t) 1
  -/
  refine eq_of_le_of_not_lt prob_le_one ?_
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ht : HasSubset.Subset s t
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.uniformOn s)
    ⊢ Not (LT.lt ((ProbabilityTheory.uniformOn s) t) 1)
  -/
  rw [not_lt, ← uniformOn_self hs hs']
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hs : s.Finite
    hs' : s.Nonempty
    ht : HasSubset.Subset s t
    this : MeasureTheory.IsProbabilityMeasure (ProbabilityTheory.uniformOn s)
    ⊢ LE.le ((ProbabilityTheory.uniformOn s) s) ((ProbabilityTheory.uniformOn s) t)
  -/
  exact measure_mono ht
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias condCount_eq_one_of := uniformOn_eq_one_of


theorem pred_true_of_uniformOn_eq_one (h : uniformOn s t = 1) : s ⊆ t := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    h : Eq ((ProbabilityTheory.uniformOn s) t) 1
    ⊢ HasSubset.Subset s t
  -/
  have hsf := finite_of_uniformOn_ne_zero (by rw [h]; exact one_ne_zero)
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    h : Eq ((ProbabilityTheory.uniformOn s) t) 1
    hsf : s.Finite
    ⊢ HasSubset.Subset s t
  -/
  rw [uniformOn, cond_apply hsf.measurableSet, mul_comm] at h
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    h : Eq (HMul.hMul (MeasureTheory.Measure.count (Inter.inter s t)) (Inv.inv (Me …
    hsf : s.Finite
    ⊢ HasSubset.Subset s t
  -/
  replace h := ENNReal.eq_inv_of_mul_eq_one_left h
  rw [inv_inv, Measure.count_apply_finite _ hsf, Measure.count_apply_finite _ (hsf.inter_of_left _),
    Nat.cast_inj] at h
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hsf : s.Finite
    h : Eq ⋯.toFinset.card hsf.toFinset.card
    ⊢ HasSubset.Subset s t
  -/
  suffices s ∩ t = s by exact this ▸ fun x hx => hx.2
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hsf : s.Finite
    h : Eq ⋯.toFinset.card hsf.toFinset.card
    ⊢ Eq (Inter.inter s t) s
  -/
  rw [← @Set.Finite.toFinset_inj _ _ _ (hsf.inter_of_left _) hsf]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t : Set Ω
    hsf : s.Finite
    h : Eq ⋯.toFinset.card hsf.toFinset.card
    ⊢ Eq ⋯.toFinset hsf.toFinset
  -/
  exact Finset.eq_of_subset_of_card_le (Set.Finite.toFinset_mono s.inter_subset_left) h.ge
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias pred_true_of_condCount_eq_one := pred_true_of_uniformOn_eq_one


theorem uniformOn_eq_zero_iff (hs : s.Finite) : uniformOn s t = 0 ↔ s ∩ t = ∅ := by
  simp [uniformOn, cond_apply hs.measurableSet, Measure.count_apply_eq_top, Set.not_infinite.2 hs,
    Measure.count_apply_finite _ (hs.inter_of_left _)]


@[deprecated (since := "2024-10-09")]
alias condCount_eq_zero_iff := uniformOn_eq_zero_iff


theorem uniformOn_of_univ (hs : s.Finite) (hs' : s.Nonempty) : uniformOn s Set.univ = 1 :=
  uniformOn_eq_one_of hs hs' s.subset_univ


@[deprecated (since := "2024-10-09")]
alias condCount_of_univ := uniformOn_of_univ


theorem uniformOn_inter (hs : s.Finite) :
    uniformOn s (t ∩ u) = uniformOn (s ∩ t) u * uniformOn s t := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t u : Set Ω
    hs : s.Finite
    ⊢ Eq ((ProbabilityTheory.uniformOn s) (Inter.inter t u)) (HMul.hMul ((Probabil …
  -/
  by_cases hst : s ∩ t = ∅
  · rw [hst, uniformOn_empty_meas, Measure.coe_zero, Pi.zero_apply, zero_mul,
      uniformOn_eq_zero_iff hs, ← Set.inter_assoc, hst, Set.empty_inter]
  rw [uniformOn, uniformOn, cond_apply hs.measurableSet, cond_apply hs.measurableSet,
    cond_apply (hs.inter_of_left _).measurableSet, mul_comm _ (Measure.count (s ∩ t)),
    ← mul_assoc, mul_comm _ (Measure.count (s ∩ t)), ← mul_assoc, ENNReal.mul_inv_cancel, one_mul,
    mul_comm, Set.inter_assoc]
    /-
      case neg.h0
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s t u : Set Ω
      hs : s.Finite
      hst : Not (Eq (Inter.inter s t) EmptyCollection.emptyCollection)
      ⊢ Ne (MeasureTheory.Measure.count (Inter.inter s t)) 0
    -/
  · rwa [← Measure.count_eq_zero_iff] at hst
    /-
      🎉 no goals
    -/
    /-
      case neg.ht
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s t u : Set Ω
      hs : s.Finite
      hst : Not (Eq (Inter.inter s t) EmptyCollection.emptyCollection)
      ⊢ Ne (MeasureTheory.Measure.count (Inter.inter s t)) Top.top
    -/
  · exact (Measure.count_apply_lt_top.2 <| hs.inter_of_left _).ne
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-10-09")]
alias condCount_inter := uniformOn_inter


theorem uniformOn_inter' (hs : s.Finite) :
    uniformOn s (t ∩ u) = uniformOn (s ∩ u) t * uniformOn s u := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t u : Set Ω
    hs : s.Finite
    ⊢ Eq ((ProbabilityTheory.uniformOn s) (Inter.inter t u)) (HMul.hMul ((Probabil …
  -/
  rw [← Set.inter_comm]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t u : Set Ω
    hs : s.Finite
    ⊢ Eq ((ProbabilityTheory.uniformOn s) (Inter.inter u t)) (HMul.hMul ((Probabil …
  -/
  exact uniformOn_inter hs
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias condCount_inter' := uniformOn_inter'


theorem uniformOn_union (hs : s.Finite) (htu : Disjoint t u) :
    uniformOn s (t ∪ u) = uniformOn s t + uniformOn s u := by
  rw [uniformOn, cond_apply hs.measurableSet, cond_apply hs.measurableSet,
    cond_apply hs.measurableSet, Set.inter_union_distrib_left, measure_union, mul_add]
  /-
    case hd
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t u : Set Ω
    hs : s.Finite
    htu : Disjoint t u
    ⊢ Disjoint (Inter.inter s t) (Inter.inter s u)
  -/
  exacts [htu.mono inf_le_right inf_le_right, (hs.inter_of_left _).measurableSet]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias condCount_union := uniformOn_union


theorem uniformOn_compl (t : Set Ω) (hs : s.Finite) (hs' : s.Nonempty) :
    uniformOn s t + uniformOn s tᶜ = 1 := by
  rw [← uniformOn_union hs disjoint_compl_right, Set.union_compl_self,
    (uniformOn_isProbabilityMeasure hs hs').measure_univ]


@[deprecated (since := "2024-10-09")]
alias condCount_compl := uniformOn_compl


theorem uniformOn_disjoint_union (hs : s.Finite) (ht : t.Finite) (hst : Disjoint s t) :
    uniformOn s u * uniformOn (s ∪ t) s + uniformOn t u * uniformOn (s ∪ t) t =
      uniformOn (s ∪ t) u := by
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s t u : Set Ω
    hs : s.Finite
    ht : t.Finite
    hst : Disjoint s t
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn s) u) ((ProbabilityTh …
  -/
  rcases s.eq_empty_or_nonempty with (rfl | hs') <;> rcases t.eq_empty_or_nonempty with (rfl | ht')
    /-
      case inl.inl
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      u : Set Ω
      hs ht : EmptyCollection.emptyCollection.Finite
      hst : Disjoint EmptyCollection.emptyCollection EmptyCollection.emptyCollection
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn EmptyCollection.empty …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inl.inr
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      t u : Set Ω
      ht : t.Finite
      hs : EmptyCollection.emptyCollection.Finite
      hst : Disjoint EmptyCollection.emptyCollection t
      ht' : t.Nonempty
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn EmptyCollection.empty …
    -/
  · simp [uniformOn_self ht ht']
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      Ω : Type u_1
      inst✝¹ : MeasurableSpace Ω
      inst✝ : MeasurableSingletonClass Ω
      s u : Set Ω
      hs : s.Finite
      hs' : s.Nonempty
      ht : EmptyCollection.emptyCollection.Finite
      hst : Disjoint s EmptyCollection.emptyCollection
      ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn s) u) ((ProbabilityTh …
    -/
  · simp [uniformOn_self hs hs']
    /-
      🎉 no goals
    -/
  rw [uniformOn, uniformOn, uniformOn, cond_apply hs.measurableSet,
    cond_apply ht.measurableSet, cond_apply (hs.union ht).measurableSet,
    cond_apply (hs.union ht).measurableSet, cond_apply (hs.union ht).measurableSet]
  conv_lhs =>
    rw [Set.union_inter_cancel_left, Set.union_inter_cancel_right,
      mul_comm (Measure.count (s ∪ t))⁻¹, mul_comm (Measure.count (s ∪ t))⁻¹, ← mul_assoc,
      ← mul_assoc, mul_comm _ (Measure.count s), mul_comm _ (Measure.count t), ← mul_assoc,
      ← mul_assoc]
  rw [ENNReal.mul_inv_cancel, ENNReal.mul_inv_cancel, one_mul, one_mul, ← add_mul, ← measure_union,
    Set.union_inter_distrib_right, mul_comm]
  exacts [hst.mono inf_le_left inf_le_left, (ht.inter_of_left _).measurableSet,
    Measure.count_ne_zero ht', (Measure.count_apply_lt_top.2 ht).ne, Measure.count_ne_zero hs',
    (Measure.count_apply_lt_top.2 hs).ne]


@[deprecated (since := "2024-10-09")]
alias condCount_disjoint_union := uniformOn_disjoint_union


/-- A version of the law of total probability for counting probabilities. -/
theorem uniformOn_add_compl_eq (u t : Set Ω) (hs : s.Finite) :
    uniformOn (s ∩ u) t * uniformOn s u + uniformOn (s ∩ uᶜ) t * uniformOn s uᶜ =
      uniformOn s t := by
  -- Porting note: The original proof used `conv_rhs`. However, that tactic timed out.
  have : uniformOn s t = (uniformOn (s ∩ u) t * uniformOn (s ∩ u ∪ s ∩ uᶜ) (s ∩ u) +
      uniformOn (s ∩ uᶜ) t * uniformOn (s ∩ u ∪ s ∩ uᶜ) (s ∩ uᶜ)) := by
    rw [uniformOn_disjoint_union (hs.inter_of_left _) (hs.inter_of_left _)
      (disjoint_compl_right.mono inf_le_right inf_le_right), Set.inter_union_compl]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s u t : Set Ω
    hs : s.Finite
    this : Eq ((ProbabilityTheory.uniformOn s) t) (HAdd.hAdd (HMul.hMul ((Probabil …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn (Inter.inter s u)) t) …
  -/
  rw [this]
  /-
    Ω : Type u_1
    inst✝¹ : MeasurableSpace Ω
    inst✝ : MeasurableSingletonClass Ω
    s u t : Set Ω
    hs : s.Finite
    this : Eq ((ProbabilityTheory.uniformOn s) t) (HAdd.hAdd (HMul.hMul ((Probabil …
    ⊢ Eq (HAdd.hAdd (HMul.hMul ((ProbabilityTheory.uniformOn (Inter.inter s u)) t) …
  -/
  simp [uniformOn_inter_self hs]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-09")]
alias condCount_add_compl_eq := uniformOn_add_compl_eq


