theorem setIntegral_congr_ae₀ (hs : NullMeasurableSet s μ) (h : ∀ᵐ x ∂μ, x ∈ s → f x = g x) :
    ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ :=
  integral_congr_ae ((ae_restrict_iff'₀ hs).2 h)


@[deprecated (since := "2024-04-17")]
alias set_integral_congr_ae₀ := setIntegral_congr_ae₀


theorem setIntegral_congr_ae (hs : MeasurableSet s) (h : ∀ᵐ x ∂μ, x ∈ s → f x = g x) :
    ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ :=
  integral_congr_ae ((ae_restrict_iff' hs).2 h)


@[deprecated (since := "2024-04-17")]
alias set_integral_congr_ae := setIntegral_congr_ae


theorem setIntegral_congr_fun₀ (hs : NullMeasurableSet s μ) (h : EqOn f g s) :
    ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ :=
  setIntegral_congr_ae₀ hs <| Eventually.of_forall h


@[deprecated (since := "2024-10-12")]
alias setIntegral_congr₀ := setIntegral_congr_fun₀


@[deprecated (since := "2024-04-17")]
alias set_integral_congr₀ := setIntegral_congr_fun₀


theorem setIntegral_congr_fun (hs : MeasurableSet s) (h : EqOn f g s) :
    ∫ x in s, f x ∂μ = ∫ x in s, g x ∂μ :=
  setIntegral_congr_ae hs <| Eventually.of_forall h


@[deprecated (since := "2024-10-12")]
alias setIntegral_congr := setIntegral_congr_fun


@[deprecated (since := "2024-04-17")]
alias set_integral_congr := setIntegral_congr_fun


theorem setIntegral_congr_set (hst : s =ᵐ[μ] t) : ∫ x in s, f x ∂μ = ∫ x in t, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    hst : (MeasureTheory.ae μ).EventuallyEq s t
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
  -/
  rw [Measure.restrict_congr_set hst]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-12")]
alias setIntegral_congr_set_ae := setIntegral_congr_set


@[deprecated (since := "2024-04-17")]
alias set_integral_congr_set_ae := setIntegral_congr_set


theorem integral_union_ae (hst : AEDisjoint μ s t) (ht : NullMeasurableSet t μ)
    (hfs : IntegrableOn f s μ) (hft : IntegrableOn f t μ) :
    ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ + ∫ x in t, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    hst : MeasureTheory.AEDisjoint μ s t
    ht : MeasureTheory.NullMeasurableSet t μ
    hfs : MeasureTheory.IntegrableOn f s μ
    hft : MeasureTheory.IntegrableOn f t μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (HAd …
  -/
  simp only [IntegrableOn, Measure.restrict_union₀ hst ht, integral_add_measure hfs hft]
  /-
    🎉 no goals
  -/


theorem setIntegral_union (hst : Disjoint s t) (ht : MeasurableSet t) (hfs : IntegrableOn f s μ)
    (hft : IntegrableOn f t μ) : ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ + ∫ x in t, f x ∂μ :=
  integral_union_ae hst.aedisjoint ht.nullMeasurableSet hfs hft


@[deprecated (since := "2024-10-12")]
alias integral_union := setIntegral_union


theorem integral_diff (ht : MeasurableSet t) (hfs : IntegrableOn f s μ) (hts : t ⊆ s) :
    ∫ x in s \ t, f x ∂μ = ∫ x in s, f x ∂μ - ∫ x in t, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasurableSet t
    hfs : MeasureTheory.IntegrableOn f s μ
    hts : HasSubset.Subset t s
    ⊢ Eq (MeasureTheory.integral (μ.restrict (SDiff.sdiff s t)) fun x => f x) (HSu …
  -/
  rw [eq_sub_iff_add_eq, ← setIntegral_union, diff_union_of_subset hts]
  /-
    case hst
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasurableSet t
    hfs : MeasureTheory.IntegrableOn f s μ
    hts : HasSubset.Subset t s
    ⊢ Disjoint (SDiff.sdiff s t) t
  -/
  exacts [disjoint_sdiff_self_left, ht, hfs.mono_set diff_subset, hfs.mono_set hts]
  /-
    🎉 no goals
  -/


theorem integral_inter_add_diff₀ (ht : NullMeasurableSet t μ) (hfs : IntegrableOn f s μ) :
    ∫ x in s ∩ t, f x ∂μ + ∫ x in s \ t, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasureTheory.NullMeasurableSet t μ
    hfs : MeasureTheory.IntegrableOn f s μ
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (Inter.inter s t)) fun x = …
  -/
  rw [← Measure.restrict_inter_add_diff₀ s ht, integral_add_measure]
    /-
      case hμ
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s t : Set X
      μ : MeasureTheory.Measure X
      ht : MeasureTheory.NullMeasurableSet t μ
      hfs : MeasureTheory.IntegrableOn f s μ
      ⊢ MeasureTheory.Integrable f (μ.restrict (Inter.inter s t))
    -/
  · exact Integrable.mono_measure hfs (Measure.restrict_mono inter_subset_left le_rfl)
    /-
      🎉 no goals
    -/
    /-
      case hν
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s t : Set X
      μ : MeasureTheory.Measure X
      ht : MeasureTheory.NullMeasurableSet t μ
      hfs : MeasureTheory.IntegrableOn f s μ
      ⊢ MeasureTheory.Integrable f (μ.restrict (SDiff.sdiff s t))
    -/
  · exact Integrable.mono_measure hfs (Measure.restrict_mono diff_subset le_rfl)
    /-
      🎉 no goals
    -/


theorem integral_inter_add_diff (ht : MeasurableSet t) (hfs : IntegrableOn f s μ) :
    ∫ x in s ∩ t, f x ∂μ + ∫ x in s \ t, f x ∂μ = ∫ x in s, f x ∂μ :=
  integral_inter_add_diff₀ ht.nullMeasurableSet hfs


theorem integral_finset_biUnion {ι : Type*} (t : Finset ι) {s : ι → Set X}
    (hs : ∀ i ∈ t, MeasurableSet (s i)) (h's : Set.Pairwise (↑t) (Disjoint on s))
    (hf : ∀ i ∈ t, IntegrableOn f (s i) μ) :
    ∫ x in ⋃ i ∈ t, s i, f x ∂μ = ∑ i ∈ t, ∫ x in s i, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    t : Finset ι
    s : ι → Set X
    hs : ∀ (i : ι), Membership.mem t i → MeasurableSet (s i)
    h's : (↑t).Pairwise (Function.onFun Disjoint s)
    hf : ∀ (i : ι), Membership.mem t i → MeasureTheory.IntegrableOn f (s i) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.iUnion fun i => Set.iUnion fun h …
  -/
  induction' t using Finset.induction_on with a t hat IH hs h's
    /-
      case empty
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      s : ι → Set X
      hs : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → MeasurableS …
      h's : (↑EmptyCollection.emptyCollection).Pairwise (Function.onFun Disjoint s)
      hf : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → MeasureTheo …
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.iUnion fun i => Set.iUnion fun h …
    -/
  · simp
    /-
      🎉 no goals
    -/
  · simp only [Finset.coe_insert, Finset.forall_mem_insert, Set.pairwise_insert,
      Finset.set_biUnion_insert] at hs hf h's ⊢
    /-
      case insert
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      s : ι → Set X
      a : ι
      t : Finset ι
      hat : Not (Membership.mem t a)
      IH : (∀ (i : ι), Membership.mem t i → MeasurableSet (s i)) → (↑t).Pairwise (Fu …
      hs : And (MeasurableSet (s a)) (∀ (x : ι), Membership.mem t x → MeasurableSet  …
      hf : And (MeasureTheory.IntegrableOn f (s a) μ) (∀ (x : ι), Membership.mem t x …
      h's : And ((↑t).Pairwise (Function.onFun Disjoint s)) (∀ (b : ι), Membership.m …
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union (s a) (Set.iUnion fun x  …
    -/
    rw [setIntegral_union _ _ hf.1 (integrableOn_finset_iUnion.2 hf.2)]
      /-
        case insert
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → E
        μ : MeasureTheory.Measure X
        ι : Type u_5
        s : ι → Set X
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        IH : (∀ (i : ι), Membership.mem t i → MeasurableSet (s i)) → (↑t).Pairwise (Fu …
        hs : And (MeasurableSet (s a)) (∀ (x : ι), Membership.mem t x → MeasurableSet  …
        hf : And (MeasureTheory.IntegrableOn f (s a) μ) (∀ (x : ι), Membership.mem t x …
        h's : And ((↑t).Pairwise (Function.onFun Disjoint s)) (∀ (b : ι), Membership.m …
        ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (s a)) fun x => f x) (Meas …
      -/
    · rw [Finset.sum_insert hat, IH hs.2 h's.1 hf.2]
      /-
        🎉 no goals
      -/
      /-
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → E
        μ : MeasureTheory.Measure X
        ι : Type u_5
        s : ι → Set X
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        IH : (∀ (i : ι), Membership.mem t i → MeasurableSet (s i)) → (↑t).Pairwise (Fu …
        hs : And (MeasurableSet (s a)) (∀ (x : ι), Membership.mem t x → MeasurableSet  …
        hf : And (MeasureTheory.IntegrableOn f (s a) μ) (∀ (x : ι), Membership.mem t x …
        h's : And ((↑t).Pairwise (Function.onFun Disjoint s)) (∀ (b : ι), Membership.m …
        ⊢ Disjoint (s a) (Set.iUnion fun i => Set.iUnion fun h => s i)
      -/
    · simp only [disjoint_iUnion_right]
      /-
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → E
        μ : MeasureTheory.Measure X
        ι : Type u_5
        s : ι → Set X
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        IH : (∀ (i : ι), Membership.mem t i → MeasurableSet (s i)) → (↑t).Pairwise (Fu …
        hs : And (MeasurableSet (s a)) (∀ (x : ι), Membership.mem t x → MeasurableSet  …
        hf : And (MeasureTheory.IntegrableOn f (s a) μ) (∀ (x : ι), Membership.mem t x …
        h's : And ((↑t).Pairwise (Function.onFun Disjoint s)) (∀ (b : ι), Membership.m …
        ⊢ ∀ (i : ι), Membership.mem t i → Disjoint (s a) (s i)
      -/
      exact fun i hi => (h's.2 i hi (ne_of_mem_of_not_mem hi hat).symm).1
      /-
        🎉 no goals
      -/
      /-
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → E
        μ : MeasureTheory.Measure X
        ι : Type u_5
        s : ι → Set X
        a : ι
        t : Finset ι
        hat : Not (Membership.mem t a)
        IH : (∀ (i : ι), Membership.mem t i → MeasurableSet (s i)) → (↑t).Pairwise (Fu …
        hs : And (MeasurableSet (s a)) (∀ (x : ι), Membership.mem t x → MeasurableSet  …
        hf : And (MeasureTheory.IntegrableOn f (s a) μ) (∀ (x : ι), Membership.mem t x …
        h's : And ((↑t).Pairwise (Function.onFun Disjoint s)) (∀ (b : ι), Membership.m …
        ⊢ MeasurableSet (Set.iUnion fun i => Set.iUnion fun h => s i)
      -/
    · exact Finset.measurableSet_biUnion _ hs.2
      /-
        🎉 no goals
      -/


theorem integral_fintype_iUnion {ι : Type*} [Fintype ι] {s : ι → Set X}
    (hs : ∀ i, MeasurableSet (s i)) (h's : Pairwise (Disjoint on s))
    (hf : ∀ i, IntegrableOn f (s i) μ) : ∫ x in ⋃ i, s i, f x ∂μ = ∑ i, ∫ x in s i, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝ : Fintype ι
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    h's : Pairwise (Function.onFun Disjoint s)
    hf : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.iUnion fun i => s i)) fun x => f …
  -/
  convert integral_finset_biUnion Finset.univ (fun i _ => hs i) _ fun i _ => hf i
    /-
      case h.e'_2.h.e'_6.h.e'_4.h.e'_3.h
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      inst✝ : Fintype ι
      s : ι → Set X
      hs : ∀ (i : ι), MeasurableSet (s i)
      h's : Pairwise (Function.onFun Disjoint s)
      hf : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
      x✝ : ι
      ⊢ Eq (s x✝) (Set.iUnion fun h => s x✝)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      E : Type u_3
      inst✝³ : MeasurableSpace X
      inst✝² : NormedAddCommGroup E
      inst✝¹ : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      inst✝ : Fintype ι
      s : ι → Set X
      hs : ∀ (i : ι), MeasurableSet (s i)
      h's : Pairwise (Function.onFun Disjoint s)
      hf : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
      ⊢ (↑Finset.univ).Pairwise (Function.onFun Disjoint s)
    -/
  · simp [pairwise_univ, h's]
    /-
      🎉 no goals
    -/


theorem setIntegral_empty : ∫ x in ∅, f x ∂μ = 0 := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ⊢ Eq (MeasureTheory.integral (μ.restrict EmptyCollection.emptyCollection) fun  …
  -/
  rw [Measure.restrict_empty, integral_zero_measure]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-12")]
alias integral_empty := setIntegral_empty


                                                                   /-
                                                                     X : Type u_1
                                                                     E : Type u_3
                                                                     inst✝² : MeasurableSpace X
                                                                     inst✝¹ : NormedAddCommGroup E
                                                                     inst✝ : NormedSpace Real E
                                                                     f : X → E
                                                                     μ : MeasureTheory.Measure X
                                                                     ⊢ Eq (MeasureTheory.integral (μ.restrict Set.univ) fun x => f x) (MeasureTheor …
                                                                   -/
theorem setIntegral_univ : ∫ x in univ, f x ∂μ = ∫ x, f x ∂μ := by rw [Measure.restrict_univ]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[deprecated (since := "2024-10-12")]
alias integral_univ := setIntegral_univ


theorem integral_add_compl₀ (hs : NullMeasurableSet s μ) (hfi : Integrable f μ) :
    ∫ x in s, f x ∂μ + ∫ x in sᶜ, f x ∂μ = ∫ x, f x ∂μ := by
  rw [
    ← integral_union_ae disjoint_compl_right.aedisjoint hs.compl hfi.integrableOn hfi.integrableOn,
    union_compl_self, setIntegral_univ]


theorem integral_add_compl (hs : MeasurableSet s) (hfi : Integrable f μ) :
    ∫ x in s, f x ∂μ + ∫ x in sᶜ, f x ∂μ = ∫ x, f x ∂μ :=
  integral_add_compl₀ hs.nullMeasurableSet hfi


theorem setIntegral_compl (hs : MeasurableSet s) (hfi : Integrable f μ) :
    ∫ x in sᶜ, f x ∂μ = ∫ x, f x ∂μ - ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    hs : MeasurableSet s
    hfi : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (HasCompl.compl s)) fun x => f x) (HS …
  -/
  rw [← integral_add_compl (μ := μ) hs hfi, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


/-- For a function `f` and a measurable set `s`, the integral of `indicator s f`
over the whole space is equal to `∫ x in s, f x ∂μ` defined as `∫ x, f x ∂(μ.restrict s)`. -/
theorem integral_indicator (hs : MeasurableSet s) :
    ∫ x, indicator s f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral μ fun x => s.indicator f x) (MeasureTheory.integr …
  -/
  by_cases hfi : IntegrableOn f s μ; swap
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s : Set X
      μ : MeasureTheory.Measure X
      hs : MeasurableSet s
      hfi : Not (MeasureTheory.IntegrableOn f s μ)
      ⊢ Eq (MeasureTheory.integral μ fun x => s.indicator f x) (MeasureTheory.integr …
    -/
  · rw [integral_undef hfi, integral_undef]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s : Set X
      μ : MeasureTheory.Measure X
      hs : MeasurableSet s
      hfi : Not (MeasureTheory.IntegrableOn f s μ)
      ⊢ Not (MeasureTheory.Integrable (s.indicator f) μ)
    -/
    rwa [integrable_indicator_iff hs]
    /-
      🎉 no goals
    -/
  calc
    ∫ x, indicator s f x ∂μ = ∫ x in s, indicator s f x ∂μ + ∫ x in sᶜ, indicator s f x ∂μ :=
      (integral_add_compl hs (hfi.integrable_indicator hs)).symm
    _ = ∫ x in s, f x ∂μ + ∫ x in sᶜ, 0 ∂μ :=
      (congr_arg₂ (· + ·) (integral_congr_ae (indicator_ae_eq_restrict hs))
        (integral_congr_ae (indicator_ae_eq_restrict_compl hs)))
    _ = ∫ x in s, f x ∂μ := by simp


theorem setIntegral_indicator (ht : MeasurableSet t) :
    ∫ x in s, t.indicator f x ∂μ = ∫ x in s ∩ t, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => t.indicator f x) (Measure …
  -/
  rw [integral_indicator ht, Measure.restrict_restrict ht, Set.inter_comm]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_indicator := setIntegral_indicator


theorem ofReal_setIntegral_one_of_measure_ne_top {X : Type*} {m : MeasurableSpace X}
    {μ : Measure X} {s : Set X} (hs : μ s ≠ ∞) : ENNReal.ofReal (∫ _ in s, (1 : ℝ) ∂μ) = μ s :=
  calc
    ENNReal.ofReal (∫ _ in s, (1 : ℝ) ∂μ) = ENNReal.ofReal (∫ _ in s, ‖(1 : ℝ)‖ ∂μ) := by
      /-
        X : Type u_5
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        s : Set X
        hs : Ne (μ s) Top.top
        ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) fun x => 1)) (ENNR …
      -/
      simp only [norm_one]
      /-
        🎉 no goals
      -/
    _ = ∫⁻ _ in s, 1 ∂μ := by
      /-
        X : Type u_5
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        s : Set X
        hs : Ne (μ s) Top.top
        ⊢ Eq (ENNReal.ofReal (MeasureTheory.integral (μ.restrict s) fun x => Norm.norm …
      -/
      rw [ofReal_integral_norm_eq_lintegral_nnnorm (integrableOn_const.2 (Or.inr hs.lt_top))]
      /-
        X : Type u_5
        m : MeasurableSpace X
        μ : MeasureTheory.Measure X
        s : Set X
        hs : Ne (μ s) Top.top
        ⊢ Eq (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(NNNorm.nnnorm 1)) (Mea …
      -/
      simp only [nnnorm_one, ENNReal.coe_one]
      /-
        🎉 no goals
      -/
    _ = μ s := setLIntegral_one _


@[deprecated (since := "2024-04-17")]
alias ofReal_set_integral_one_of_measure_ne_top := ofReal_setIntegral_one_of_measure_ne_top


theorem ofReal_setIntegral_one {X : Type*} {_ : MeasurableSpace X} (μ : Measure X)
    [IsFiniteMeasure μ] (s : Set X) : ENNReal.ofReal (∫ _ in s, (1 : ℝ) ∂μ) = μ s :=
  ofReal_setIntegral_one_of_measure_ne_top (measure_ne_top μ s)


@[deprecated (since := "2024-04-17")]
alias ofReal_set_integral_one := ofReal_setIntegral_one


theorem integral_piecewise [DecidablePred (· ∈ s)] (hs : MeasurableSet s) (hf : IntegrableOn f s μ)
    (hg : IntegrableOn g sᶜ μ) :
    ∫ x, s.piecewise f g x ∂μ = ∫ x in s, f x ∂μ + ∫ x in sᶜ, g x ∂μ := by
  rw [← Set.indicator_add_compl_eq_piecewise,
    integral_add' (hf.integrable_indicator hs) (hg.integrable_indicator hs.compl),
    integral_indicator hs, integral_indicator hs.compl]


theorem tendsto_setIntegral_of_monotone
    {ι : Type*} [Preorder ι] [(atTop : Filter ι).IsCountablyGenerated]
    {s : ι → Set X} (hsm : ∀ i, MeasurableSet (s i)) (h_mono : Monotone s)
    (hfi : IntegrableOn f (⋃ n, s n) μ) :
    Tendsto (fun i => ∫ x in s i, f x ∂μ) atTop (𝓝 (∫ x in ⋃ n, s n, f x ∂μ)) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hfi : MeasureTheory.IntegrableOn f (Set.iUnion fun n => s n) μ
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  refine .of_neBot_imp fun hne ↦ ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hfi : MeasureTheory.IntegrableOn f (Set.iUnion fun n => s n) μ
    hne : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  have := (atTop_neBot_iff.mp hne).2
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hfi : MeasureTheory.IntegrableOn f (Set.iUnion fun n => s n) μ
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  have hfi' : ∫⁻ x in ⋃ n, s n, ‖f x‖₊ ∂μ < ∞ := hfi.2
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hfi : MeasureTheory.IntegrableOn f (Set.iUnion fun n => s n) μ
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    hfi' : LT.lt (MeasureTheory.lintegral (μ.restrict (Set.iUnion fun n => s n)) f …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  set S := ⋃ i, s i
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hfi' : LT.lt (MeasureTheory.lintegral (μ.restrict S) fun x => ↑(NNNorm.nnnorm  …
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  have hSm : MeasurableSet S := MeasurableSet.iUnion_of_monotone h_mono hsm
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hfi' : LT.lt (MeasureTheory.lintegral (μ.restrict S) fun x => ↑(NNNorm.nnnorm  …
    hSm : MeasurableSet S
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  have hsub {i} : s i ⊆ S := subset_iUnion s i
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hfi' : LT.lt (MeasureTheory.lintegral (μ.restrict S) fun x => ↑(NNNorm.nnnorm  …
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  rw [← withDensity_apply _ hSm] at hfi'
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hfi' : LT.lt ((μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))) S) Top.top
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  set ν := μ.withDensity fun x => ‖f x‖₊ with hν
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ν : MeasureTheory.Measure X := μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))
    hfi' : LT.lt (ν S) Top.top
    hν : Eq ν (μ.withDensity fun x => ↑(NNNorm.nnnorm (f x)))
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  refine Metric.nhds_basis_closedBall.tendsto_right_iff.2 fun ε ε0 => ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ν : MeasureTheory.Measure X := μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))
    hfi' : LT.lt (ν S) Top.top
    hν : Eq ν (μ.withDensity fun x => ↑(NNNorm.nnnorm (f x)))
    ε : Real
    ε0 : LT.lt 0 ε
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.closedBall (MeasureTheory …
  -/
  lift ε to ℝ≥0 using ε0.le
  have : ∀ᶠ i in atTop, ν (s i) ∈ Icc (ν S - ε) (ν S + ε) :=
    tendsto_measure_iUnion_atTop h_mono (ENNReal.Icc_mem_nhds hfi'.ne (ENNReal.coe_pos.2 ε0).ne')
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ν : MeasureTheory.Measure X := μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))
    hfi' : LT.lt (ν S) Top.top
    hν : Eq ν (μ.withDensity fun x => ↑(NNNorm.nnnorm (f x)))
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    this : Filter.Eventually (fun i => Membership.mem (Set.Icc (HSub.hSub (ν S) ↑ε …
    ⊢ Filter.Eventually (fun x => Membership.mem (Metric.closedBall (MeasureTheory …
  -/
  filter_upwards [this] with i hi
  rw [mem_closedBall_iff_norm', ← integral_diff (hsm i) hfi hsub, ← coe_nnnorm, NNReal.coe_le_coe, ←
    ENNReal.coe_le_coe]
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ν : MeasureTheory.Measure X := μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))
    hfi' : LT.lt (ν S) Top.top
    hν : Eq ν (μ.withDensity fun x => ↑(NNNorm.nnnorm (f x)))
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    this : Filter.Eventually (fun i => Membership.mem (Set.Icc (HSub.hSub (ν S) ↑ε …
    i : ι
    hi : Membership.mem (Set.Icc (HSub.hSub (ν S) ↑ε) (HAdd.hAdd (ν S) ↑ε)) (ν (s  …
    ⊢ LE.le ↑(NNNorm.nnnorm (MeasureTheory.integral (μ.restrict (SDiff.sdiff S (s  …
  -/
  refine (ennnorm_integral_le_lintegral_ennnorm _).trans ?_
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_mono : Monotone s
    hne : Filter.atTop.NeBot
    this✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
    S : Set X := Set.iUnion fun i => s i
    hfi : MeasureTheory.IntegrableOn f S μ
    hSm : MeasurableSet S
    hsub : ∀ {i : ι}, HasSubset.Subset (s i) S
    ν : MeasureTheory.Measure X := μ.withDensity fun x => ↑(NNNorm.nnnorm (f x))
    hfi' : LT.lt (ν S) Top.top
    hν : Eq ν (μ.withDensity fun x => ↑(NNNorm.nnnorm (f x)))
    ε : NNReal
    ε0 : LT.lt 0 ↑ε
    this : Filter.Eventually (fun i => Membership.mem (Set.Icc (HSub.hSub (ν S) ↑ε …
    i : ι
    hi : Membership.mem (Set.Icc (HSub.hSub (ν S) ↑ε) (HAdd.hAdd (ν S) ↑ε)) (ν (s  …
    ⊢ LE.le (MeasureTheory.lintegral (μ.restrict (SDiff.sdiff S (s i))) fun a => ↑ …
  -/
  rw [← withDensity_apply _ (hSm.diff (hsm _)), ← hν, measure_diff hsub (hsm _).nullMeasurableSet]
  exacts [tsub_le_iff_tsub_le.mp hi.1,
    (hi.2.trans_lt <| ENNReal.add_lt_top.2 ⟨hfi', ENNReal.coe_lt_top⟩).ne]


@[deprecated (since := "2024-04-17")]
alias tendsto_set_integral_of_monotone := tendsto_setIntegral_of_monotone


theorem tendsto_setIntegral_of_antitone
    {ι : Type*} [Preorder ι] [(atTop : Filter ι).IsCountablyGenerated]
    {s : ι → Set X} (hsm : ∀ i, MeasurableSet (s i)) (h_anti : Antitone s)
    (hfi : ∃ i, IntegrableOn f (s i) μ) :
    Tendsto (fun i ↦ ∫ x in s i, f x ∂μ) atTop (𝓝 (∫ x in ⋂ n, s n, f x ∂μ)) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : Exists fun i => MeasureTheory.IntegrableOn f (s i) μ
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  refine .of_neBot_imp fun hne ↦ ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : Exists fun i => MeasureTheory.IntegrableOn f (s i) μ
    hne : Filter.atTop.NeBot
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  have := (atTop_neBot_iff.mp hne).2
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_anti : Antitone s
    hfi : Exists fun i => MeasureTheory.IntegrableOn f (s i) μ
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    ⊢ Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (s i)) fun x =>  …
  -/
  rcases hfi with ⟨i₀, hi₀⟩
  suffices Tendsto (∫ x in s i₀, f x ∂μ - ∫ x in s i₀ \ s ·, f x ∂μ) atTop
      (𝓝 (∫ x in s i₀, f x ∂μ - ∫ x in ⋃ i, s i₀ \ s i, f x ∂μ)) by
    convert this.congr' <| (eventually_ge_atTop i₀).mono fun i hi ↦ ?_
    · rw [← diff_iInter, integral_diff _ hi₀ (iInter_subset _ _), sub_sub_cancel]
      exact .iInter_of_antitone h_anti hsm
    · rw [integral_diff (hsm i) hi₀ (h_anti hi), sub_sub_cancel]
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_anti : Antitone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    i₀ : ι
    hi₀ : MeasureTheory.IntegrableOn f (s i₀) μ
    ⊢ Filter.Tendsto (fun x => HSub.hSub (MeasureTheory.integral (μ.restrict (s i₀ …
  -/
  apply tendsto_const_nhds.sub
  /-
    case intro
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝¹ : Preorder ι
    inst✝ : Filter.atTop.IsCountablyGenerated
    s : ι → Set X
    hsm : ∀ (i : ι), MeasurableSet (s i)
    h_anti : Antitone s
    hne : Filter.atTop.NeBot
    this : IsDirected ι fun x1 x2 => LE.le x1 x2
    i₀ : ι
    hi₀ : MeasureTheory.IntegrableOn f (s i₀) μ
    ⊢ Filter.Tendsto (fun x => MeasureTheory.integral (μ.restrict (SDiff.sdiff (s  …
  -/
  refine tendsto_setIntegral_of_monotone (by measurability) ?_ ?_
    /-
      case intro.refine_1
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      inst✝¹ : Preorder ι
      inst✝ : Filter.atTop.IsCountablyGenerated
      s : ι → Set X
      hsm : ∀ (i : ι), MeasurableSet (s i)
      h_anti : Antitone s
      hne : Filter.atTop.NeBot
      this : IsDirected ι fun x1 x2 => LE.le x1 x2
      i₀ : ι
      hi₀ : MeasureTheory.IntegrableOn f (s i₀) μ
      ⊢ Monotone fun x => SDiff.sdiff (s i₀) (s x)
    -/
  · exact fun i j h ↦ diff_subset_diff_right (h_anti h)
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      inst✝¹ : Preorder ι
      inst✝ : Filter.atTop.IsCountablyGenerated
      s : ι → Set X
      hsm : ∀ (i : ι), MeasurableSet (s i)
      h_anti : Antitone s
      hne : Filter.atTop.NeBot
      this : IsDirected ι fun x1 x2 => LE.le x1 x2
      i₀ : ι
      hi₀ : MeasureTheory.IntegrableOn f (s i₀) μ
      ⊢ MeasureTheory.IntegrableOn f (Set.iUnion fun n => SDiff.sdiff (s i₀) (s n)) μ
    -/
  · rw [← diff_iInter]
    /-
      case intro.refine_2
      X : Type u_1
      E : Type u_3
      inst✝⁴ : MeasurableSpace X
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace Real E
      f : X → E
      μ : MeasureTheory.Measure X
      ι : Type u_5
      inst✝¹ : Preorder ι
      inst✝ : Filter.atTop.IsCountablyGenerated
      s : ι → Set X
      hsm : ∀ (i : ι), MeasurableSet (s i)
      h_anti : Antitone s
      hne : Filter.atTop.NeBot
      this : IsDirected ι fun x1 x2 => LE.le x1 x2
      i₀ : ι
      hi₀ : MeasureTheory.IntegrableOn f (s i₀) μ
      ⊢ MeasureTheory.IntegrableOn f (SDiff.sdiff (s i₀) (Set.iInter fun i => s i)) μ
    -/
    exact hi₀.mono_set diff_subset
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias tendsto_set_integral_of_antitone := tendsto_setIntegral_of_antitone


theorem hasSum_integral_iUnion_ae {ι : Type*} [Countable ι] {s : ι → Set X}
    (hm : ∀ i, NullMeasurableSet (s i) μ) (hd : Pairwise (AEDisjoint μ on s))
    (hfi : IntegrableOn f (⋃ i, s i) μ) :
    HasSum (fun n => ∫ x in s n, f x ∂μ) (∫ x in ⋃ n, s n, f x ∂μ) := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝ : Countable ι
    s : ι → Set X
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    hfi : MeasureTheory.IntegrableOn f (Set.iUnion fun i => s i) μ
    ⊢ HasSum (fun n => MeasureTheory.integral (μ.restrict (s n)) fun x => f x) (Me …
  -/
  simp only [IntegrableOn, Measure.restrict_iUnion_ae hd hm] at hfi ⊢
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    ι : Type u_5
    inst✝ : Countable ι
    s : ι → Set X
    hm : ∀ (i : ι), MeasureTheory.NullMeasurableSet (s i) μ
    hd : Pairwise (Function.onFun (MeasureTheory.AEDisjoint μ) s)
    hfi : MeasureTheory.Integrable f (MeasureTheory.Measure.sum fun i => μ.restric …
    ⊢ HasSum (fun n => MeasureTheory.integral (μ.restrict (s n)) fun x => f x) (Me …
  -/
  exact hasSum_integral_measure hfi
  /-
    🎉 no goals
  -/


theorem hasSum_integral_iUnion {ι : Type*} [Countable ι] {s : ι → Set X}
    (hm : ∀ i, MeasurableSet (s i)) (hd : Pairwise (Disjoint on s))
    (hfi : IntegrableOn f (⋃ i, s i) μ) :
    HasSum (fun n => ∫ x in s n, f x ∂μ) (∫ x in ⋃ n, s n, f x ∂μ) :=
  hasSum_integral_iUnion_ae (fun i => (hm i).nullMeasurableSet) (hd.mono fun _ _ h => h.aedisjoint)
    hfi


theorem integral_iUnion {ι : Type*} [Countable ι] {s : ι → Set X} (hm : ∀ i, MeasurableSet (s i))
    (hd : Pairwise (Disjoint on s)) (hfi : IntegrableOn f (⋃ i, s i) μ) :
    ∫ x in ⋃ n, s n, f x ∂μ = ∑' n, ∫ x in s n, f x ∂μ :=
  (HasSum.tsum_eq (hasSum_integral_iUnion hm hd hfi)).symm


theorem integral_iUnion_ae {ι : Type*} [Countable ι] {s : ι → Set X}
    (hm : ∀ i, NullMeasurableSet (s i) μ) (hd : Pairwise (AEDisjoint μ on s))
    (hfi : IntegrableOn f (⋃ i, s i) μ) : ∫ x in ⋃ n, s n, f x ∂μ = ∑' n, ∫ x in s n, f x ∂μ :=
  (HasSum.tsum_eq (hasSum_integral_iUnion_ae hm hd hfi)).symm


theorem setIntegral_eq_zero_of_ae_eq_zero (ht_eq : ∀ᵐ x ∂μ, x ∈ t → f x = 0) :
    ∫ x in t, f x ∂μ = 0 := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) 0
  -/
  by_cases hf : AEStronglyMeasurable f (μ.restrict t); swap
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      t : Set X
      μ : MeasureTheory.Measure X
      ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
      hf : Not (MeasureTheory.AEStronglyMeasurable f (μ.restrict t))
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) 0
    -/
  · rw [integral_undef]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      t : Set X
      μ : MeasureTheory.Measure X
      ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
      hf : Not (MeasureTheory.AEStronglyMeasurable f (μ.restrict t))
      ⊢ Not (MeasureTheory.Integrable f (μ.restrict t))
    -/
    contrapose! hf
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      t : Set X
      μ : MeasureTheory.Measure X
      ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
      hf : MeasureTheory.Integrable f (μ.restrict t)
      ⊢ MeasureTheory.AEStronglyMeasurable f (μ.restrict t)
    -/
    exact hf.1
    /-
      🎉 no goals
    -/
  have : ∫ x in t, hf.mk f x ∂μ = 0 := by
    refine integral_eq_zero_of_ae ?_
    rw [EventuallyEq,
      ae_restrict_iff (hf.stronglyMeasurable_mk.measurableSet_eq_fun stronglyMeasurable_zero)]
    filter_upwards [ae_imp_of_ae_restrict hf.ae_eq_mk, ht_eq] with x hx h'x h''x
    rw [← hx h''x]
    exact h'x h''x
  /-
    case pos
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict t)
    this : Eq (MeasureTheory.integral (μ.restrict t) fun x => MeasureTheory.AEStro …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) 0
  -/
  rw [← this]
  /-
    case pos
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Membership.mem t x → Eq (f x) 0) (MeasureT …
    hf : MeasureTheory.AEStronglyMeasurable f (μ.restrict t)
    this : Eq (MeasureTheory.integral (μ.restrict t) fun x => MeasureTheory.AEStro …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  exact integral_congr_ae hf.ae_eq_mk
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_zero_of_ae_eq_zero := setIntegral_eq_zero_of_ae_eq_zero


theorem setIntegral_eq_zero_of_forall_eq_zero (ht_eq : ∀ x ∈ t, f x = 0) :
    ∫ x in t, f x ∂μ = 0 :=
  setIntegral_eq_zero_of_ae_eq_zero (Eventually.of_forall ht_eq)


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_zero_of_forall_eq_zero := setIntegral_eq_zero_of_forall_eq_zero


theorem integral_union_eq_left_of_ae_aux (ht_eq : ∀ᵐ x ∂μ.restrict t, f x = 0)
    (haux : StronglyMeasurable f) (H : IntegrableOn f (s ∪ t) μ) :
    ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  let k := f ⁻¹' {0}
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  have hk : MeasurableSet k := by borelize E; exact haux.measurable (measurableSet_singleton _)
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  have h's : IntegrableOn f s μ := H.mono subset_union_left le_rfl
  have A : ∀ u : Set X, ∫ x in u ∩ k, f x ∂μ = 0 := fun u =>
    setIntegral_eq_zero_of_forall_eq_zero fun x hx => hx.2
  rw [← integral_inter_add_diff hk h's, ← integral_inter_add_diff hk H, A, A, zero_add, zero_add,
    union_diff_distrib, union_comm]
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    h's : MeasureTheory.IntegrableOn f s μ
    A : ∀ (u : Set X), Eq (MeasureTheory.integral (μ.restrict (Inter.inter u k)) f …
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union (SDiff.sdiff t k) (SDiff …
  -/
  apply setIntegral_congr_set
  /-
    case hst
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    h's : MeasureTheory.IntegrableOn f s μ
    A : ∀ (u : Set X), Eq (MeasureTheory.integral (μ.restrict (Inter.inter u k)) f …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (Union.union (SDiff.sdiff t k) (SDiff.sdif …
  -/
  rw [union_ae_eq_right]
  /-
    case hst
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    h's : MeasureTheory.IntegrableOn f s μ
    A : ∀ (u : Set X), Eq (MeasureTheory.integral (μ.restrict (Inter.inter u k)) f …
    ⊢ Eq (μ (SDiff.sdiff (SDiff.sdiff t k) (SDiff.sdiff s k))) 0
  -/
  apply measure_mono_null diff_subset
  /-
    case hst
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    h's : MeasureTheory.IntegrableOn f s μ
    A : ∀ (u : Set X), Eq (MeasureTheory.integral (μ.restrict (Inter.inter u k)) f …
    ⊢ Eq (μ (SDiff.sdiff t k)) 0
  -/
  rw [measure_zero_iff_ae_nmem]
  /-
    case hst
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    haux : MeasureTheory.StronglyMeasurable f
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    hk : MeasurableSet k
    h's : MeasureTheory.IntegrableOn f s μ
    A : ∀ (u : Set X), Eq (MeasureTheory.integral (μ.restrict (Inter.inter u k)) f …
    ⊢ Filter.Eventually (fun a => Not (Membership.mem (SDiff.sdiff t k) a)) (Measu …
  -/
  filter_upwards [ae_imp_of_ae_restrict ht_eq] with x hx h'x using h'x.2 (hx h'x.1)
  /-
    🎉 no goals
  -/


theorem integral_union_eq_left_of_ae (ht_eq : ∀ᵐ x ∂μ.restrict t, f x = 0) :
    ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  have ht : IntegrableOn f t μ := by apply integrableOn_zero.congr_fun_ae; symm; exact ht_eq
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    ht : MeasureTheory.IntegrableOn f t μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  by_cases H : IntegrableOn f (s ∪ t) μ; swap
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s t : Set X
      μ : MeasureTheory.Measure X
      ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
      ht : MeasureTheory.IntegrableOn f t μ
      H : Not (MeasureTheory.IntegrableOn f (Union.union s t) μ)
      ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
    -/
  · rw [integral_undef H, integral_undef]; simpa [integrableOn_union, ht] using H
                                           /-
                                             🎉 no goals
                                           -/
  /-
    case pos
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht_eq : Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict  …
    ht : MeasureTheory.IntegrableOn f t μ
    H : MeasureTheory.IntegrableOn f (Union.union s t) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union s t)) fun x => f x) (Mea …
  -/
  let f' := H.1.mk f
  calc
    ∫ x : X in s ∪ t, f x ∂μ = ∫ x : X in s ∪ t, f' x ∂μ := integral_congr_ae H.1.ae_eq_mk
    _ = ∫ x in s, f' x ∂μ := by
      apply
        integral_union_eq_left_of_ae_aux _ H.1.stronglyMeasurable_mk (H.congr_fun_ae H.1.ae_eq_mk)
      filter_upwards [ht_eq,
        ae_mono (Measure.restrict_mono subset_union_right le_rfl) H.1.ae_eq_mk] with x hx h'x
      rw [← h'x, hx]
    _ = ∫ x in s, f x ∂μ :=
      integral_congr_ae
        (ae_mono (Measure.restrict_mono subset_union_left le_rfl) H.1.ae_eq_mk.symm)


theorem integral_union_eq_left_of_forall₀ {f : X → E} (ht : NullMeasurableSet t μ)
    (ht_eq : ∀ x ∈ t, f x = 0) : ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ :=
  integral_union_eq_left_of_ae ((ae_restrict_iff'₀ ht).2 (Eventually.of_forall ht_eq))


theorem integral_union_eq_left_of_forall {f : X → E} (ht : MeasurableSet t)
    (ht_eq : ∀ x ∈ t, f x = 0) : ∫ x in s ∪ t, f x ∂μ = ∫ x in s, f x ∂μ :=
  integral_union_eq_left_of_forall₀ ht.nullMeasurableSet ht_eq


theorem setIntegral_eq_of_subset_of_ae_diff_eq_zero_aux (hts : s ⊆ t)
    (h't : ∀ᵐ x ∂μ, x ∈ t \ s → f x = 0) (haux : StronglyMeasurable f)
    (h'aux : IntegrableOn f t μ) : ∫ x in t, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    hts : HasSubset.Subset s t
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    haux : MeasureTheory.StronglyMeasurable f
    h'aux : MeasureTheory.IntegrableOn f t μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  let k := f ⁻¹' {0}
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    hts : HasSubset.Subset s t
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    haux : MeasureTheory.StronglyMeasurable f
    h'aux : MeasureTheory.IntegrableOn f t μ
    k : Set X := Set.preimage f (Singleton.singleton 0)
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  have hk : MeasurableSet k := by borelize E; exact haux.measurable (measurableSet_singleton _)
  calc
    ∫ x in t, f x ∂μ = ∫ x in t ∩ k, f x ∂μ + ∫ x in t \ k, f x ∂μ := by
      rw [integral_inter_add_diff hk h'aux]
    _ = ∫ x in t \ k, f x ∂μ := by
      rw [setIntegral_eq_zero_of_forall_eq_zero fun x hx => ?_, zero_add]; exact hx.2
    _ = ∫ x in s \ k, f x ∂μ := by
      apply setIntegral_congr_set
      filter_upwards [h't] with x hx
      change (x ∈ t \ k) = (x ∈ s \ k)
      simp only [mem_preimage, mem_singleton_iff, eq_iff_iff, and_congr_left_iff, mem_diff]
      intro h'x
      by_cases xs : x ∈ s
      · simp only [xs, hts xs]
      · simp only [xs, iff_false]
        intro xt
        exact h'x (hx ⟨xt, xs⟩)
    _ = ∫ x in s ∩ k, f x ∂μ + ∫ x in s \ k, f x ∂μ := by
      have : ∀ x ∈ s ∩ k, f x = 0 := fun x hx => hx.2
      rw [setIntegral_eq_zero_of_forall_eq_zero this, zero_add]
    _ = ∫ x in s, f x ∂μ := by rw [integral_inter_add_diff hk (h'aux.mono hts le_rfl)]


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_of_subset_of_ae_diff_eq_zero_aux :=
  setIntegral_eq_of_subset_of_ae_diff_eq_zero_aux


/-- If a function vanishes almost everywhere on `t \ s` with `s ⊆ t`, then its integrals on `s`
and `t` coincide if `t` is null-measurable. -/
theorem setIntegral_eq_of_subset_of_ae_diff_eq_zero (ht : NullMeasurableSet t μ) (hts : s ⊆ t)
    (h't : ∀ᵐ x ∂μ, x ∈ t \ s → f x = 0) : ∫ x in t, f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasureTheory.NullMeasurableSet t μ
    hts : HasSubset.Subset s t
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  by_cases h : IntegrableOn f t μ; swap
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s t : Set X
      μ : MeasureTheory.Measure X
      ht : MeasureTheory.NullMeasurableSet t μ
      hts : HasSubset.Subset s t
      h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
      h : Not (MeasureTheory.IntegrableOn f t μ)
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
    -/
  · have : ¬IntegrableOn f s μ := fun H => h (H.of_ae_diff_eq_zero ht h't)
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s t : Set X
      μ : MeasureTheory.Measure X
      ht : MeasureTheory.NullMeasurableSet t μ
      hts : HasSubset.Subset s t
      h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
      h : Not (MeasureTheory.IntegrableOn f t μ)
      this : Not (MeasureTheory.IntegrableOn f s μ)
      ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
    -/
    rw [integral_undef h, integral_undef this]
    /-
      🎉 no goals
    -/
  /-
    case pos
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s t : Set X
    μ : MeasureTheory.Measure X
    ht : MeasureTheory.NullMeasurableSet t μ
    hts : HasSubset.Subset s t
    h't : Filter.Eventually (fun x => Membership.mem (SDiff.sdiff t s) x → Eq (f x …
    h : MeasureTheory.IntegrableOn f t μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict t) fun x => f x) (MeasureTheory.integ …
  -/
  let f' := h.1.mk f
  calc
    ∫ x in t, f x ∂μ = ∫ x in t, f' x ∂μ := integral_congr_ae h.1.ae_eq_mk
    _ = ∫ x in s, f' x ∂μ := by
      apply
        setIntegral_eq_of_subset_of_ae_diff_eq_zero_aux hts _ h.1.stronglyMeasurable_mk
          (h.congr h.1.ae_eq_mk)
      filter_upwards [h't, ae_imp_of_ae_restrict h.1.ae_eq_mk] with x hx h'x h''x
      rw [← h'x h''x.1, hx h''x]
    _ = ∫ x in s, f x ∂μ := by
      apply integral_congr_ae
      apply ae_restrict_of_ae_restrict_of_subset hts
      exact h.1.ae_eq_mk.symm


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_of_subset_of_ae_diff_eq_zero := setIntegral_eq_of_subset_of_ae_diff_eq_zero


/-- If a function vanishes on `t \ s` with `s ⊆ t`, then its integrals on `s`
and `t` coincide if `t` is measurable. -/
theorem setIntegral_eq_of_subset_of_forall_diff_eq_zero (ht : MeasurableSet t) (hts : s ⊆ t)
    (h't : ∀ x ∈ t \ s, f x = 0) : ∫ x in t, f x ∂μ = ∫ x in s, f x ∂μ :=
  setIntegral_eq_of_subset_of_ae_diff_eq_zero ht.nullMeasurableSet hts
    (Eventually.of_forall fun x hx => h't x hx)


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_of_subset_of_forall_diff_eq_zero :=
  setIntegral_eq_of_subset_of_forall_diff_eq_zero


/-- If a function vanishes almost everywhere on `sᶜ`, then its integral on `s`
coincides with its integral on the whole space. -/
theorem setIntegral_eq_integral_of_ae_compl_eq_zero (h : ∀ᵐ x ∂μ, x ∉ s → f x = 0) :
    ∫ x in s, f x ∂μ = ∫ x, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    h : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Measur …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
  -/
  symm
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    h : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Measur …
    ⊢ Eq (MeasureTheory.integral μ fun x => f x) (MeasureTheory.integral (μ.restri …
  -/
  nth_rw 1 [← setIntegral_univ]
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    h : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Measur …
    ⊢ Eq (MeasureTheory.integral (μ.restrict Set.univ) fun x => f x) (MeasureTheor …
  -/
  apply setIntegral_eq_of_subset_of_ae_diff_eq_zero nullMeasurableSet_univ (subset_univ _)
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    h : Filter.Eventually (fun x => Not (Membership.mem s x) → Eq (f x) 0) (Measur …
    ⊢ Filter.Eventually (fun x => Membership.mem (SDiff.sdiff Set.univ s) x → Eq ( …
  -/
  filter_upwards [h] with x hx h'x using hx h'x.2
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_integral_of_ae_compl_eq_zero := setIntegral_eq_integral_of_ae_compl_eq_zero


/-- If a function vanishes on `sᶜ`, then its integral on `s` coincides with its integral on the
whole space. -/
theorem setIntegral_eq_integral_of_forall_compl_eq_zero (h : ∀ x, x ∉ s → f x = 0) :
    ∫ x in s, f x ∂μ = ∫ x, f x ∂μ :=
  setIntegral_eq_integral_of_ae_compl_eq_zero (Eventually.of_forall h)


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_integral_of_forall_compl_eq_zero :=
  setIntegral_eq_integral_of_forall_compl_eq_zero


theorem setIntegral_neg_eq_setIntegral_nonpos [LinearOrder E] {f : X → E}
    (hf : AEStronglyMeasurable f μ) :
    ∫ x in {x | f x < 0}, f x ∂μ = ∫ x in {x | f x ≤ 0}, f x ∂μ := by
  have h_union : {x | f x ≤ 0} = {x | f x < 0} ∪ {x | f x = 0} := by
    simp_rw [le_iff_lt_or_eq, setOf_or]
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    inst✝ : LinearOrder E
    f : X → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_union : Eq (setOf fun x => LE.le (f x) 0) (Union.union (setOf fun x => LT.lt …
    ⊢ Eq (MeasureTheory.integral (μ.restrict (setOf fun x => LT.lt (f x) 0)) fun x …
  -/
  rw [h_union]
  have B : NullMeasurableSet {x | f x = 0} μ :=
    hf.nullMeasurableSet_eq_fun aestronglyMeasurable_zero
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    inst✝ : LinearOrder E
    f : X → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_union : Eq (setOf fun x => LE.le (f x) 0) (Union.union (setOf fun x => LT.lt …
    B : MeasureTheory.NullMeasurableSet (setOf fun x => Eq (f x) 0) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (setOf fun x => LT.lt (f x) 0)) fun x …
  -/
  symm
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    inst✝ : LinearOrder E
    f : X → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_union : Eq (setOf fun x => LE.le (f x) 0) (Union.union (setOf fun x => LT.lt …
    B : MeasureTheory.NullMeasurableSet (setOf fun x => Eq (f x) 0) μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Union.union (setOf fun x => LT.lt (f …
  -/
  refine integral_union_eq_left_of_ae ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    inst✝ : LinearOrder E
    f : X → E
    hf : MeasureTheory.AEStronglyMeasurable f μ
    h_union : Eq (setOf fun x => LE.le (f x) 0) (Union.union (setOf fun x => LT.lt …
    B : MeasureTheory.NullMeasurableSet (setOf fun x => Eq (f x) 0) μ
    ⊢ Filter.Eventually (fun x => Eq (f x) 0) (MeasureTheory.ae (μ.restrict (setOf …
  -/
  filter_upwards [ae_restrict_mem₀ B] with x hx using hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_neg_eq_set_integral_nonpos := setIntegral_neg_eq_setIntegral_nonpos


theorem integral_norm_eq_pos_sub_neg {f : X → ℝ} (hfi : Integrable f μ) :
    ∫ x, ‖f x‖ ∂μ = ∫ x in {x | 0 ≤ f x}, f x ∂μ - ∫ x in {x | f x ≤ 0}, f x ∂μ :=
  have h_meas : NullMeasurableSet {x | 0 ≤ f x} μ :=
    aestronglyMeasurable_const.nullMeasurableSet_le hfi.1
  calc
    ∫ x, ‖f x‖ ∂μ = ∫ x in {x | 0 ≤ f x}, ‖f x‖ ∂μ + ∫ x in {x | 0 ≤ f x}ᶜ, ‖f x‖ ∂μ := by
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (MeasureTheory.integral μ fun x => Norm.norm (f x)) (HAdd.hAdd (MeasureTh …
      -/
      rw [← integral_add_compl₀ h_meas hfi.norm]
      /-
        🎉 no goals
      -/
    _ = ∫ x in {x | 0 ≤ f x}, f x ∂μ + ∫ x in {x | 0 ≤ f x}ᶜ, ‖f x‖ ∂μ := by
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (setOf fun x => LE.le 0 (f …
      -/
      congr 1
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (MeasureTheory.integral (μ.restrict (setOf fun x => LE.le 0 (f x))) fun x …
      -/
      refine setIntegral_congr_fun₀ h_meas fun x hx => ?_
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (setOf fun x => LE.le 0 (f x)) x
        ⊢ Eq (Norm.norm (f x)) (f x)
      -/
      dsimp only
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (setOf fun x => LE.le 0 (f x)) x
        ⊢ Eq (Norm.norm (f x)) (f x)
      -/
      rw [Real.norm_eq_abs, abs_eq_self.mpr _]
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (setOf fun x => LE.le 0 (f x)) x
        ⊢ LE.le 0 (f x)
      -/
      exact hx
      /-
        🎉 no goals
      -/
    _ = ∫ x in {x | 0 ≤ f x}, f x ∂μ - ∫ x in {x | 0 ≤ f x}ᶜ, f x ∂μ := by
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (HAdd.hAdd (MeasureTheory.integral (μ.restrict (setOf fun x => LE.le 0 (f …
      -/
      congr 1
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (MeasureTheory.integral (μ.restrict (HasCompl.compl (setOf fun x => LE.le …
      -/
      rw [← integral_neg]
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (MeasureTheory.integral (μ.restrict (HasCompl.compl (setOf fun x => LE.le …
      -/
      refine setIntegral_congr_fun₀ h_meas.compl fun x hx => ?_
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (HasCompl.compl (setOf fun x => LE.le 0 (f x))) x
        ⊢ Eq (Norm.norm (f x)) (Neg.neg (f x))
      -/
      dsimp only
      /-
        case e_a
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (HasCompl.compl (setOf fun x => LE.le 0 (f x))) x
        ⊢ Eq (Norm.norm (f x)) (Neg.neg (f x))
      -/
      rw [Real.norm_eq_abs, abs_eq_neg_self.mpr _]
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Membership.mem (HasCompl.compl (setOf fun x => LE.le 0 (f x))) x
        ⊢ LE.le (f x) 0
      -/
      rw [Set.mem_compl_iff, Set.nmem_setOf_iff] at hx
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        x : X
        hx : Not (LE.le 0 (f x))
        ⊢ LE.le (f x) 0
      -/
      linarith
      /-
        🎉 no goals
      -/
    _ = ∫ x in {x | 0 ≤ f x}, f x ∂μ - ∫ x in {x | f x ≤ 0}, f x ∂μ := by
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        hfi : MeasureTheory.Integrable f μ
        h_meas : MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (f x)) μ
        ⊢ Eq (HSub.hSub (MeasureTheory.integral (μ.restrict (setOf fun x => LE.le 0 (f …
      -/
      rw [← setIntegral_neg_eq_setIntegral_nonpos hfi.1, compl_setOf]; simp only [not_le]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem setIntegral_const [CompleteSpace E] (c : E) : ∫ _ in s, c ∂μ = (μ s).toReal • c := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    s : Set X
    μ : MeasureTheory.Measure X
    inst✝ : CompleteSpace E
    c : E
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => c) (HSMul.hSMul (μ s).toR …
  -/
  rw [integral_const, Measure.restrict_apply_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_const := setIntegral_const


@[simp]
theorem integral_indicator_const [CompleteSpace E] (e : E) ⦃s : Set X⦄ (s_meas : MeasurableSet s) :
    ∫ x : X, s.indicator (fun _ : X => e) x ∂μ = (μ s).toReal • e := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    inst✝ : CompleteSpace E
    e : E
    s : Set X
    s_meas : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral μ fun x => s.indicator (fun x => e) x) (HSMul.hSM …
  -/
  rw [integral_indicator s_meas, ← setIntegral_const]
  /-
    🎉 no goals
  -/


@[simp]
theorem integral_indicator_one ⦃s : Set X⦄ (hs : MeasurableSet s) :
    ∫ x, s.indicator 1 x ∂μ = (μ s).toReal :=
  (integral_indicator_const 1 hs).trans ((smul_eq_mul _).trans (mul_one _))


theorem setIntegral_indicatorConstLp [CompleteSpace E]
    {p : ℝ≥0∞} (hs : MeasurableSet s) (ht : MeasurableSet t) (hμt : μ t ≠ ∞) (e : E) :
    ∫ x in s, indicatorConstLp p ht hμt e x ∂μ = (μ (t ∩ s)).toReal • e :=
  calc
    ∫ x in s, indicatorConstLp p ht hμt e x ∂μ = ∫ x in s, t.indicator (fun _ => e) x ∂μ := by
      /-
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        s t : Set X
        μ : MeasureTheory.Measure X
        inst✝ : CompleteSpace E
        p : ENNReal
        hs : MeasurableSet s
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        e : E
        ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑(MeasureTheory.indicato …
      -/
      rw [setIntegral_congr_ae hs (indicatorConstLp_coeFn.mono fun x hx _ => hx)]
      /-
        🎉 no goals
      -/
                                     /-
                                       X : Type u_1
                                       E : Type u_3
                                       inst✝³ : MeasurableSpace X
                                       inst✝² : NormedAddCommGroup E
                                       inst✝¹ : NormedSpace Real E
                                       s t : Set X
                                       μ : MeasureTheory.Measure X
                                       inst✝ : CompleteSpace E
                                       p : ENNReal
                                       hs : MeasurableSet s
                                       ht : MeasurableSet t
                                       hμt : Ne (μ t) Top.top
                                       e : E
                                       ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => t.indicator (fun x => e)  …
                                     -/
    _ = (μ (t ∩ s)).toReal • e := by rw [integral_indicator_const _ ht, Measure.restrict_apply ht]
                                     /-
                                       🎉 no goals
                                     -/


@[deprecated (since := "2024-04-17")]
alias set_integral_indicatorConstLp := setIntegral_indicatorConstLp


theorem integral_indicatorConstLp [CompleteSpace E]
    {p : ℝ≥0∞} (ht : MeasurableSet t) (hμt : μ t ≠ ∞) (e : E) :
    ∫ x, indicatorConstLp p ht hμt e x ∂μ = (μ t).toReal • e :=
  calc
    ∫ x, indicatorConstLp p ht hμt e x ∂μ = ∫ x in univ, indicatorConstLp p ht hμt e x ∂μ := by
      /-
        X : Type u_1
        E : Type u_3
        inst✝³ : MeasurableSpace X
        inst✝² : NormedAddCommGroup E
        inst✝¹ : NormedSpace Real E
        t : Set X
        μ : MeasureTheory.Measure X
        inst✝ : CompleteSpace E
        p : ENNReal
        ht : MeasurableSet t
        hμt : Ne (μ t) Top.top
        e : E
        ⊢ Eq (MeasureTheory.integral μ fun x => ↑↑(MeasureTheory.indicatorConstLp p ht …
      -/
      rw [setIntegral_univ]
      /-
        🎉 no goals
      -/
    _ = (μ (t ∩ univ)).toReal • e := setIntegral_indicatorConstLp MeasurableSet.univ ht hμt e
                               /-
                                 X : Type u_1
                                 E : Type u_3
                                 inst✝³ : MeasurableSpace X
                                 inst✝² : NormedAddCommGroup E
                                 inst✝¹ : NormedSpace Real E
                                 t : Set X
                                 μ : MeasureTheory.Measure X
                                 inst✝ : CompleteSpace E
                                 p : ENNReal
                                 ht : MeasurableSet t
                                 hμt : Ne (μ t) Top.top
                                 e : E
                                 ⊢ Eq (HSMul.hSMul (μ (Inter.inter t Set.univ)).toReal e) (HSMul.hSMul (μ t).to …
                               -/
    _ = (μ t).toReal • e := by rw [inter_univ]
                               /-
                                 🎉 no goals
                               -/


theorem setIntegral_map {Y} [MeasurableSpace Y] {g : X → Y} {f : Y → E} {s : Set Y}
    (hs : MeasurableSet s) (hf : AEStronglyMeasurable f (Measure.map g μ)) (hg : AEMeasurable g μ) :
    ∫ y in s, f y ∂Measure.map g μ = ∫ x in g ⁻¹' s, f (g x) ∂μ := by
  rw [Measure.restrict_map_of_aemeasurable hg hs,
    integral_map (hg.mono_measure Measure.restrict_le_self) (hf.mono_measure _)]
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    Y : Type u_5
    inst✝ : MeasurableSpace Y
    g : X → Y
    f : Y → E
    s : Set Y
    hs : MeasurableSet s
    hf : MeasureTheory.AEStronglyMeasurable f (MeasureTheory.Measure.map g μ)
    hg : AEMeasurable g μ
    ⊢ LE.le (MeasureTheory.Measure.map g (μ.restrict (Set.preimage g s))) (Measure …
  -/
  exact Measure.map_mono_of_aemeasurable Measure.restrict_le_self hg
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_map := setIntegral_map


theorem _root_.MeasurableEmbedding.setIntegral_map {Y} {_ : MeasurableSpace Y} {f : X → Y}
    (hf : MeasurableEmbedding f) (g : Y → E) (s : Set Y) :
    ∫ y in s, g y ∂Measure.map f μ = ∫ x in f ⁻¹' s, g (f x) ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    μ : MeasureTheory.Measure X
    Y : Type u_5
    x✝ : MeasurableSpace Y
    f : X → Y
    hf : MeasurableEmbedding f
    g : Y → E
    s : Set Y
    ⊢ Eq (MeasureTheory.integral ((MeasureTheory.Measure.map f μ).restrict s) fun  …
  -/
  rw [hf.restrict_map, hf.integral_map]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias _root_.MeasurableEmbedding.set_integral_map := _root_.MeasurableEmbedding.setIntegral_map


theorem _root_.Topology.IsClosedEmbedding.setIntegral_map [TopologicalSpace X] [BorelSpace X] {Y}
    [MeasurableSpace Y] [TopologicalSpace Y] [BorelSpace Y] {g : X → Y} {f : Y → E} (s : Set Y)
    (hg : IsClosedEmbedding g) : ∫ y in s, f y ∂Measure.map g μ = ∫ x in g ⁻¹' s, f (g x) ∂μ :=
  hg.measurableEmbedding.setIntegral_map _ _


@[deprecated (since := "2024-10-20")]
alias _root_.ClosedEmbedding.setIntegral_map := IsClosedEmbedding.setIntegral_map


@[deprecated (since := "2024-04-17")]
alias _root_.IsClosedEmbedding.set_integral_map :=
  IsClosedEmbedding.setIntegral_map


@[deprecated (since := "2024-10-20")]
alias _root_.ClosedEmbedding.set_integral_map := IsClosedEmbedding.set_integral_map


theorem MeasurePreserving.setIntegral_preimage_emb {Y} {_ : MeasurableSpace Y} {f : X → Y} {ν}
    (h₁ : MeasurePreserving f μ ν) (h₂ : MeasurableEmbedding f) (g : Y → E) (s : Set Y) :
    ∫ x in f ⁻¹' s, g (f x) ∂μ = ∫ y in s, g y ∂ν :=
  (h₁.restrict_preimage_emb h₂ s).integral_comp h₂ _


@[deprecated (since := "2024-04-17")]
alias MeasurePreserving.set_integral_preimage_emb := MeasurePreserving.setIntegral_preimage_emb


theorem MeasurePreserving.setIntegral_image_emb {Y} {_ : MeasurableSpace Y} {f : X → Y} {ν}
    (h₁ : MeasurePreserving f μ ν) (h₂ : MeasurableEmbedding f) (g : Y → E) (s : Set X) :
    ∫ y in f '' s, g y ∂ν = ∫ x in s, g (f x) ∂μ :=
  Eq.symm <| (h₁.restrict_image_emb h₂ s).integral_comp h₂ _


@[deprecated (since := "2024-04-17")]
alias MeasurePreserving.set_integral_image_emb := MeasurePreserving.setIntegral_image_emb


theorem setIntegral_map_equiv {Y} [MeasurableSpace Y] (e : X ≃ᵐ Y) (f : Y → E) (s : Set Y) :
    ∫ y in s, f y ∂Measure.map e μ = ∫ x in e ⁻¹' s, f (e x) ∂μ :=
  e.measurableEmbedding.setIntegral_map f s


@[deprecated (since := "2024-04-17")]
alias set_integral_map_equiv := setIntegral_map_equiv


theorem norm_setIntegral_le_of_norm_le_const_ae {C : ℝ} (hs : μ s < ∞)
    (hC : ∀ᵐ x ∂μ.restrict s, ‖f x‖ ≤ C) : ‖∫ x in s, f x ∂μ‖ ≤ C * (μ s).toReal := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt (μ s) Top.top
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae  …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun x => f x)) (HMul …
  -/
  rw [← Measure.restrict_apply_univ] at *
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt ((μ.restrict s) Set.univ) Top.top
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae  …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun x => f x)) (HMul …
  -/
  haveI : IsFiniteMeasure (μ.restrict s) := ⟨hs⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt ((μ.restrict s) Set.univ) Top.top
    hC : Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae  …
    this : MeasureTheory.IsFiniteMeasure (μ.restrict s)
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun x => f x)) (HMul …
  -/
  exact norm_integral_le_of_norm_le_const hC
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias norm_set_integral_le_of_norm_le_const_ae := norm_setIntegral_le_of_norm_le_const_ae


theorem norm_setIntegral_le_of_norm_le_const_ae' {C : ℝ} (hs : μ s < ∞)
    (hC : ∀ᵐ x ∂μ, x ∈ s → ‖f x‖ ≤ C) (hfm : AEStronglyMeasurable f (μ.restrict s)) :
    ‖∫ x in s, f x ∂μ‖ ≤ C * (μ s).toReal := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt (μ s) Top.top
    hC : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x))  …
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun x => f x)) (HMul …
  -/
  apply norm_setIntegral_le_of_norm_le_const_ae hs
  have A : ∀ᵐ x : X ∂μ, x ∈ s → ‖AEStronglyMeasurable.mk f hfm x‖ ≤ C := by
    filter_upwards [hC, hfm.ae_mem_imp_eq_mk] with _ h1 h2 h3
    rw [← h2 h3]
    exact h1 h3
  have B : MeasurableSet {x | ‖hfm.mk f x‖ ≤ C} :=
    hfm.stronglyMeasurable_mk.norm.measurable measurableSet_Iic
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt (μ s) Top.top
    hC : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x))  …
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    A : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (Measure …
    B : MeasurableSet (setOf fun x => LE.le (Norm.norm (MeasureTheory.AEStronglyMe …
    ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae (μ. …
  -/
  filter_upwards [hfm.ae_eq_mk, (ae_restrict_iff B).2 A] with _ h1 _
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → E
    s : Set X
    μ : MeasureTheory.Measure X
    C : Real
    hs : LT.lt (μ s) Top.top
    hC : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x))  …
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    A : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (Measure …
    B : MeasurableSet (setOf fun x => LE.le (Norm.norm (MeasureTheory.AEStronglyMe …
    a✝¹ : X
    h1 : Eq (f a✝¹) (MeasureTheory.AEStronglyMeasurable.mk f hfm a✝¹)
    a✝ : LE.le (Norm.norm (MeasureTheory.AEStronglyMeasurable.mk f hfm a✝¹)) C
    ⊢ LE.le (Norm.norm (f a✝¹)) C
  -/
  rwa [h1]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias norm_set_integral_le_of_norm_le_const_ae' := norm_setIntegral_le_of_norm_le_const_ae'


theorem norm_setIntegral_le_of_norm_le_const_ae'' {C : ℝ} (hs : μ s < ∞) (hsm : MeasurableSet s)
    (hC : ∀ᵐ x ∂μ, x ∈ s → ‖f x‖ ≤ C) : ‖∫ x in s, f x ∂μ‖ ≤ C * (μ s).toReal :=
  norm_setIntegral_le_of_norm_le_const_ae hs <| by
    /-
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → E
      s : Set X
      μ : MeasureTheory.Measure X
      C : Real
      hs : LT.lt (μ s) Top.top
      hsm : MeasurableSet s
      hC : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (f x))  …
      ⊢ Filter.Eventually (fun x => LE.le (Norm.norm (f x)) C) (MeasureTheory.ae (μ. …
    -/
    rwa [ae_restrict_eq hsm, eventually_inf_principal]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias norm_set_integral_le_of_norm_le_const_ae'' := norm_setIntegral_le_of_norm_le_const_ae''


theorem norm_setIntegral_le_of_norm_le_const {C : ℝ} (hs : μ s < ∞) (hC : ∀ x ∈ s, ‖f x‖ ≤ C)
    (hfm : AEStronglyMeasurable f (μ.restrict s)) : ‖∫ x in s, f x ∂μ‖ ≤ C * (μ s).toReal :=
  norm_setIntegral_le_of_norm_le_const_ae' hs (Eventually.of_forall hC) hfm


@[deprecated (since := "2024-04-17")]
alias norm_set_integral_le_of_norm_le_const := norm_setIntegral_le_of_norm_le_const


theorem norm_setIntegral_le_of_norm_le_const' {C : ℝ} (hs : μ s < ∞) (hsm : MeasurableSet s)
    (hC : ∀ x ∈ s, ‖f x‖ ≤ C) : ‖∫ x in s, f x ∂μ‖ ≤ C * (μ s).toReal :=
  norm_setIntegral_le_of_norm_le_const_ae'' hs hsm <| Eventually.of_forall hC


@[deprecated (since := "2024-04-17")]
alias norm_set_integral_le_of_norm_le_const' := norm_setIntegral_le_of_norm_le_const'


theorem setIntegral_eq_zero_iff_of_nonneg_ae {f : X → ℝ} (hf : 0 ≤ᵐ[μ.restrict s] f)
    (hfi : IntegrableOn f s μ) : ∫ x in s, f x ∂μ = 0 ↔ f =ᵐ[μ.restrict s] 0 :=
  integral_eq_zero_iff_of_nonneg_ae hf hfi


@[deprecated (since := "2024-04-17")]
alias set_integral_eq_zero_iff_of_nonneg_ae := setIntegral_eq_zero_iff_of_nonneg_ae


theorem setIntegral_pos_iff_support_of_nonneg_ae {f : X → ℝ} (hf : 0 ≤ᵐ[μ.restrict s] f)
    (hfi : IntegrableOn f s μ) : (0 < ∫ x in s, f x ∂μ) ↔ 0 < μ (support f ∩ s) := by
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    s : Set X
    μ : MeasureTheory.Measure X
    f : X → Real
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 f
    hfi : MeasureTheory.IntegrableOn f s μ
    ⊢ Iff (LT.lt 0 (MeasureTheory.integral (μ.restrict s) fun x => f x)) (LT.lt 0  …
  -/
  rw [integral_pos_iff_support_of_nonneg_ae hf hfi, Measure.restrict_apply₀]
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    s : Set X
    μ : MeasureTheory.Measure X
    f : X → Real
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 f
    hfi : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.NullMeasurableSet (Function.support f) (μ.restrict s)
  -/
  rw [support_eq_preimage]
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    s : Set X
    μ : MeasureTheory.Measure X
    f : X → Real
    hf : (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 f
    hfi : MeasureTheory.IntegrableOn f s μ
    ⊢ MeasureTheory.NullMeasurableSet (Set.preimage f (HasCompl.compl (Singleton.s …
  -/
  exact hfi.aestronglyMeasurable.aemeasurable.nullMeasurable (measurableSet_singleton 0).compl
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_pos_iff_support_of_nonneg_ae := setIntegral_pos_iff_support_of_nonneg_ae


theorem setIntegral_gt_gt {R : ℝ} {f : X → ℝ} (hR : 0 ≤ R)
    (hfint : IntegrableOn f {x | ↑R < f x} μ) (hμ : μ {x | ↑R < f x} ≠ 0) :
    (μ {x | ↑R < f x}).toReal * R < ∫ x in {x | ↑R < f x}, f x ∂μ := by
  have : IntegrableOn (fun _ => R) {x | ↑R < f x} μ := by
    refine ⟨aestronglyMeasurable_const, lt_of_le_of_lt ?_ hfint.2⟩
    refine setLIntegral_mono_ae hfint.1.ennnorm <| ae_of_all _ fun x hx => ?_
    simp only [ENNReal.coe_le_coe, Real.nnnorm_of_nonneg hR, enorm_eq_nnnorm,
      Real.nnnorm_of_nonneg (hR.trans <| le_of_lt hx), Subtype.mk_le_mk]
    exact le_of_lt hx
  rw [← sub_pos, ← smul_eq_mul, ← setIntegral_const, ← integral_sub hfint this,
    setIntegral_pos_iff_support_of_nonneg_ae]
    /-
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      R : Real
      f : X → Real
      hR : LE.le 0 R
      hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
      hμ : Ne (μ (setOf fun x => LT.lt R (f x))) 0
      this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
      ⊢ LT.lt 0 (μ (Inter.inter (Function.support fun a => HSub.hSub (f a) R) (setOf …
    -/
  · rw [← zero_lt_iff] at hμ
    /-
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      R : Real
      f : X → Real
      hR : LE.le 0 R
      hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
      hμ : LT.lt 0 (μ (setOf fun x => LT.lt R (f x)))
      this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
      ⊢ LT.lt 0 (μ (Inter.inter (Function.support fun a => HSub.hSub (f a) R) (setOf …
    -/
    rwa [Set.inter_eq_self_of_subset_right]
    /-
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      R : Real
      f : X → Real
      hR : LE.le 0 R
      hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
      hμ : LT.lt 0 (μ (setOf fun x => LT.lt R (f x)))
      this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
      ⊢ HasSubset.Subset (setOf fun x => LT.lt R (f x)) (Function.support fun a => H …
    -/
    exact fun x hx => Ne.symm (ne_of_lt <| sub_pos.2 hx)
    /-
      🎉 no goals
    -/
    /-
      case hf
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      R : Real
      f : X → Real
      hR : LE.le 0 R
      hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
      hμ : Ne (μ (setOf fun x => LT.lt R (f x))) 0
      this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
      ⊢ (MeasureTheory.ae (μ.restrict (setOf fun x => LT.lt R (f x)))).EventuallyLE  …
    -/
  · rw [Pi.zero_def, EventuallyLE, ae_restrict_iff₀]
      /-
        case hf
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        R : Real
        f : X → Real
        hR : LE.le 0 R
        hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
        hμ : Ne (μ (setOf fun x => LT.lt R (f x))) 0
        this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
        ⊢ Filter.Eventually (fun x => Membership.mem (setOf fun x => LT.lt R (f x)) x  …
      -/
    · exact Eventually.of_forall fun x hx => sub_nonneg.2 <| le_of_lt hx
      /-
        🎉 no goals
      -/
      /-
        case hf
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        R : Real
        f : X → Real
        hR : LE.le 0 R
        hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
        hμ : Ne (μ (setOf fun x => LT.lt R (f x))) 0
        this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
        ⊢ MeasureTheory.NullMeasurableSet (setOf fun x => LE.le 0 (HSub.hSub (f x) R)) …
      -/
    · exact nullMeasurableSet_le aemeasurable_zero (hfint.1.aemeasurable.sub aemeasurable_const)
      /-
        🎉 no goals
      -/
    /-
      case hfi
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      R : Real
      f : X → Real
      hR : LE.le 0 R
      hfint : MeasureTheory.IntegrableOn f (setOf fun x => LT.lt R (f x)) μ
      hμ : Ne (μ (setOf fun x => LT.lt R (f x))) 0
      this : MeasureTheory.IntegrableOn (fun x => R) (setOf fun x => LT.lt R (f x)) μ
      ⊢ MeasureTheory.IntegrableOn (fun a => HSub.hSub (f a) R) (setOf fun x => LT.l …
    -/
  · exact Integrable.sub hfint this
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_gt_gt := setIntegral_gt_gt


theorem setIntegral_trim {X} {m m0 : MeasurableSpace X} {μ : Measure X} (hm : m ≤ m0) {f : X → E}
    (hf_meas : StronglyMeasurable[m] f) {s : Set X} (hs : MeasurableSet[m] s) :
    ∫ x in s, f x ∂μ = ∫ x in s, f x ∂μ.trim hm := by
  /-
    E : Type u_3
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    X : Type u_5
    m m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    hm : LE.le m m0
    f : X → E
    hf_meas : MeasureTheory.StronglyMeasurable f
    s : Set X
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.integ …
  -/
  rwa [integral_trim hm hf_meas, restrict_trim hm μ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_trim := setIntegral_trim


theorem integral_Icc_eq_integral_Ioc' (hx : μ {x} = 0) :
    ∫ t in Icc x y, f t ∂μ = ∫ t in Ioc x y, f t ∂μ :=
  setIntegral_congr_set (Ioc_ae_eq_Icc' hx).symm


theorem integral_Icc_eq_integral_Ico' (hy : μ {y} = 0) :
    ∫ t in Icc x y, f t ∂μ = ∫ t in Ico x y, f t ∂μ :=
  setIntegral_congr_set (Ico_ae_eq_Icc' hy).symm


theorem integral_Ioc_eq_integral_Ioo' (hy : μ {y} = 0) :
    ∫ t in Ioc x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ :=
  setIntegral_congr_set (Ioo_ae_eq_Ioc' hy).symm


theorem integral_Ico_eq_integral_Ioo' (hx : μ {x} = 0) :
    ∫ t in Ico x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ :=
  setIntegral_congr_set (Ioo_ae_eq_Ico' hx).symm


theorem integral_Icc_eq_integral_Ioo' (hx : μ {x} = 0) (hy : μ {y} = 0) :
    ∫ t in Icc x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ :=
  setIntegral_congr_set (Ioo_ae_eq_Icc' hx hy).symm


theorem integral_Iic_eq_integral_Iio' (hx : μ {x} = 0) :
    ∫ t in Iic x, f t ∂μ = ∫ t in Iio x, f t ∂μ :=
  setIntegral_congr_set (Iio_ae_eq_Iic' hx).symm


theorem integral_Ici_eq_integral_Ioi' (hx : μ {x} = 0) :
    ∫ t in Ici x, f t ∂μ = ∫ t in Ioi x, f t ∂μ :=
  setIntegral_congr_set (Ioi_ae_eq_Ici' hx).symm


theorem integral_Icc_eq_integral_Ioc : ∫ t in Icc x y, f t ∂μ = ∫ t in Ioc x y, f t ∂μ :=
  integral_Icc_eq_integral_Ioc' <| measure_singleton x


theorem integral_Icc_eq_integral_Ico : ∫ t in Icc x y, f t ∂μ = ∫ t in Ico x y, f t ∂μ :=
  integral_Icc_eq_integral_Ico' <| measure_singleton y


theorem integral_Ioc_eq_integral_Ioo : ∫ t in Ioc x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ :=
  integral_Ioc_eq_integral_Ioo' <| measure_singleton y


theorem integral_Ico_eq_integral_Ioo : ∫ t in Ico x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ :=
  integral_Ico_eq_integral_Ioo' <| measure_singleton x


theorem integral_Icc_eq_integral_Ioo : ∫ t in Icc x y, f t ∂μ = ∫ t in Ioo x y, f t ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    f : X → E
    μ : MeasureTheory.Measure X
    inst✝¹ : PartialOrder X
    x y : X
    inst✝ : MeasureTheory.NoAtoms μ
    ⊢ Eq (MeasureTheory.integral (μ.restrict (Set.Icc x y)) fun t => f t) (Measure …
  -/
  rw [integral_Icc_eq_integral_Ico, integral_Ico_eq_integral_Ioo]
  /-
    🎉 no goals
  -/


theorem integral_Iic_eq_integral_Iio : ∫ t in Iic x, f t ∂μ = ∫ t in Iio x, f t ∂μ :=
  integral_Iic_eq_integral_Iio' <| measure_singleton x


theorem integral_Ici_eq_integral_Ioi : ∫ t in Ici x, f t ∂μ = ∫ t in Ioi x, f t ∂μ :=
  integral_Ici_eq_integral_Ioi' <| measure_singleton x


theorem setIntegral_mono_ae_restrict (h : f ≤ᵐ[μ.restrict s] g) :
    ∫ x in s, f x ∂μ ≤ ∫ x in s, g x ∂μ :=
  integral_mono_ae hf hg h


@[deprecated (since := "2024-04-17")]
alias set_integral_mono_ae_restrict := setIntegral_mono_ae_restrict


theorem setIntegral_mono_ae (h : f ≤ᵐ[μ] g) : ∫ x in s, f x ∂μ ≤ ∫ x in s, g x ∂μ :=
  setIntegral_mono_ae_restrict hf hg (ae_restrict_of_ae h)


@[deprecated (since := "2024-04-17")]
alias set_integral_mono_ae := setIntegral_mono_ae


theorem setIntegral_mono_on (hs : MeasurableSet s) (h : ∀ x ∈ s, f x ≤ g x) :
    ∫ x in s, f x ∂μ ≤ ∫ x in s, g x ∂μ :=
  setIntegral_mono_ae_restrict hf hg
        /-
          X : Type u_1
          inst✝ : MeasurableSpace X
          μ : MeasureTheory.Measure X
          f g : X → Real
          s : Set X
          hf : MeasureTheory.IntegrableOn f s μ
          hg : MeasureTheory.IntegrableOn g s μ
          hs : MeasurableSet s
          h : ∀ (x : X), Membership.mem s x → LE.le (f x) (g x)
          ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE f g
        -/
    (by simp [hs, EventuallyLE, eventually_inf_principal, ae_of_all _ h])
        /-
          🎉 no goals
        -/


@[deprecated (since := "2024-04-17")]
alias set_integral_mono_on := setIntegral_mono_on


theorem setIntegral_mono_on_ae (hs : MeasurableSet s) (h : ∀ᵐ x ∂μ, x ∈ s → f x ≤ g x) :
    ∫ x in s, f x ∂μ ≤ ∫ x in s, g x ∂μ := by
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f g : X → Real
    s : Set X
    hf : MeasureTheory.IntegrableOn f s μ
    hg : MeasureTheory.IntegrableOn g s μ
    hs : MeasurableSet s
    h : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) (g x)) (Measu …
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => f x) (MeasureTheory.in …
  -/
  refine setIntegral_mono_ae_restrict hf hg ?_; rwa [EventuallyLE, ae_restrict_iff' hs]
                                                /-
                                                  🎉 no goals
                                                -/


@[deprecated (since := "2024-04-17")]
alias set_integral_mono_on_ae := setIntegral_mono_on_ae


theorem setIntegral_mono (h : f ≤ g) : ∫ x in s, f x ∂μ ≤ ∫ x in s, g x ∂μ :=
  integral_mono hf hg h


@[deprecated (since := "2024-04-17")]
alias set_integral_mono := setIntegral_mono


theorem setIntegral_mono_set (hfi : IntegrableOn f t μ) (hf : 0 ≤ᵐ[μ.restrict t] f)
    (hst : s ≤ᵐ[μ] t) : ∫ x in s, f x ∂μ ≤ ∫ x in t, f x ∂μ :=
  integral_mono_measure (Measure.restrict_mono_ae hst) hf hfi


@[deprecated (since := "2024-04-17")]
alias set_integral_mono_set := setIntegral_mono_set


theorem setIntegral_le_integral (hfi : Integrable f μ) (hf : 0 ≤ᵐ[μ] f) :
    ∫ x in s, f x ∂μ ≤ ∫ x, f x ∂μ :=
  integral_mono_measure (Measure.restrict_le_self) hf hfi


@[deprecated (since := "2024-04-17")]
alias set_integral_le_integral := setIntegral_le_integral


theorem setIntegral_ge_of_const_le {c : ℝ} (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (hf : ∀ x ∈ s, c ≤ f x) (hfint : IntegrableOn (fun x : X => f x) s μ) :
    c * (μ s).toReal ≤ ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    c : Real
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    hf : ∀ (x : X), Membership.mem s x → LE.le c (f x)
    hfint : MeasureTheory.IntegrableOn (fun x => f x) s μ
    ⊢ LE.le (HMul.hMul c (μ s).toReal) (MeasureTheory.integral (μ.restrict s) fun  …
  -/
  rw [mul_comm, ← smul_eq_mul, ← setIntegral_const c]
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    c : Real
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    hf : ∀ (x : X), Membership.mem s x → LE.le c (f x)
    hfint : MeasureTheory.IntegrableOn (fun x => f x) s μ
    ⊢ LE.le (MeasureTheory.integral (μ.restrict s) fun x => c) (MeasureTheory.inte …
  -/
  exact setIntegral_mono_on (integrableOn_const.2 (Or.inr hμs.lt_top)) hfint hs hf
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_ge_of_const_le := setIntegral_ge_of_const_le


theorem setIntegral_nonneg_of_ae_restrict (hf : 0 ≤ᵐ[μ.restrict s] f) : 0 ≤ ∫ x in s, f x ∂μ :=
  integral_nonneg_of_ae hf


@[deprecated (since := "2024-04-17")]
alias set_integral_nonneg_of_ae_restrict := setIntegral_nonneg_of_ae_restrict


theorem setIntegral_nonneg_of_ae (hf : 0 ≤ᵐ[μ] f) : 0 ≤ ∫ x in s, f x ∂μ :=
  setIntegral_nonneg_of_ae_restrict (ae_restrict_of_ae hf)


@[deprecated (since := "2024-04-17")]
alias set_integral_nonneg_of_ae := setIntegral_nonneg_of_ae


theorem setIntegral_nonneg (hs : MeasurableSet s) (hf : ∀ x, x ∈ s → 0 ≤ f x) :
    0 ≤ ∫ x in s, f x ∂μ :=
  setIntegral_nonneg_of_ae_restrict ((ae_restrict_iff' hs).mpr (ae_of_all μ hf))


@[deprecated (since := "2024-04-17")]
alias set_integral_nonneg := setIntegral_nonneg


theorem setIntegral_nonneg_ae (hs : MeasurableSet s) (hf : ∀ᵐ x ∂μ, x ∈ s → 0 ≤ f x) :
    0 ≤ ∫ x in s, f x ∂μ :=
                                          /-
                                            X : Type u_1
                                            inst✝ : MeasurableSpace X
                                            μ : MeasureTheory.Measure X
                                            f : X → Real
                                            s : Set X
                                            hs : MeasurableSet s
                                            hf : Filter.Eventually (fun x => Membership.mem s x → LE.le 0 (f x)) (MeasureT …
                                            ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE 0 f
                                          -/
  setIntegral_nonneg_of_ae_restrict <| by rwa [EventuallyLE, ae_restrict_iff' hs]
                                          /-
                                            🎉 no goals
                                          -/


@[deprecated (since := "2024-04-17")]
alias set_integral_nonneg_ae := setIntegral_nonneg_ae


theorem setIntegral_le_nonneg {s : Set X} (hs : MeasurableSet s) (hf : StronglyMeasurable f)
    (hfi : Integrable f μ) : ∫ x in s, f x ∂μ ≤ ∫ x in {y | 0 ≤ f y}, f x ∂μ := by
  rw [← integral_indicator hs, ←
    integral_indicator (stronglyMeasurable_const.measurableSet_le hf)]
  exact
    integral_mono (hfi.indicator hs)
      (hfi.indicator (stronglyMeasurable_const.measurableSet_le hf))
      (indicator_le_indicator_nonneg s f)


@[deprecated (since := "2024-04-17")]
alias set_integral_le_nonneg := setIntegral_le_nonneg


theorem setIntegral_nonpos_of_ae_restrict (hf : f ≤ᵐ[μ.restrict s] 0) : ∫ x in s, f x ∂μ ≤ 0 :=
  integral_nonpos_of_ae hf


@[deprecated (since := "2024-04-17")]
alias set_integral_nonpos_of_ae_restrict := setIntegral_nonpos_of_ae_restrict


theorem setIntegral_nonpos_of_ae (hf : f ≤ᵐ[μ] 0) : ∫ x in s, f x ∂μ ≤ 0 :=
  setIntegral_nonpos_of_ae_restrict (ae_restrict_of_ae hf)


@[deprecated (since := "2024-04-17")]
alias set_integral_nonpos_of_ae := setIntegral_nonpos_of_ae


theorem setIntegral_nonpos_ae (hs : MeasurableSet s) (hf : ∀ᵐ x ∂μ, x ∈ s → f x ≤ 0) :
    ∫ x in s, f x ∂μ ≤ 0 :=
                                          /-
                                            X : Type u_1
                                            inst✝ : MeasurableSpace X
                                            μ : MeasureTheory.Measure X
                                            f : X → Real
                                            s : Set X
                                            hs : MeasurableSet s
                                            hf : Filter.Eventually (fun x => Membership.mem s x → LE.le (f x) 0) (MeasureT …
                                            ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyLE f 0
                                          -/
  setIntegral_nonpos_of_ae_restrict <| by rwa [EventuallyLE, ae_restrict_iff' hs]
                                          /-
                                            🎉 no goals
                                          -/


@[deprecated (since := "2024-04-17")]
alias set_integral_nonpos_ae := setIntegral_nonpos_ae


theorem setIntegral_nonpos (hs : MeasurableSet s) (hf : ∀ x, x ∈ s → f x ≤ 0) :
    ∫ x in s, f x ∂μ ≤ 0 :=
  setIntegral_nonpos_ae hs <| ae_of_all μ hf


@[deprecated (since := "2024-04-17")]
alias set_integral_nonpos := setIntegral_nonpos


theorem setIntegral_nonpos_le {s : Set X} (hs : MeasurableSet s) (hf : StronglyMeasurable f)
    (hfi : Integrable f μ) : ∫ x in {y | f y ≤ 0}, f x ∂μ ≤ ∫ x in s, f x ∂μ := by
  rw [← integral_indicator hs, ←
    integral_indicator (hf.measurableSet_le stronglyMeasurable_const)]
  exact
    integral_mono (hfi.indicator (hf.measurableSet_le stronglyMeasurable_const))
      (hfi.indicator hs) (indicator_nonpos_le_indicator s f)


@[deprecated (since := "2024-04-17")]
alias set_integral_nonpos_le := setIntegral_nonpos_le


lemma Integrable.measure_le_integral {f : X → ℝ} (f_int : Integrable f μ) (f_nonneg : 0 ≤ᵐ[μ] f)
    {s : Set X} (hs : ∀ x ∈ s, 1 ≤ f x) :
    μ s ≤ ENNReal.ofReal (∫ x, f x ∂μ) := by
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    f_int : MeasureTheory.Integrable f μ
    f_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le 1 (f x)
    ⊢ LE.le (μ s) (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x))
  -/
  rw [ofReal_integral_eq_lintegral_ofReal f_int f_nonneg]
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    f_int : MeasureTheory.Integrable f μ
    f_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le 1 (f x)
    ⊢ LE.le (μ s) (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (f x))
  -/
  apply meas_le_lintegral₀
    /-
      case hf
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      f_int : MeasureTheory.Integrable f μ
      f_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le 1 (f x)
      ⊢ AEMeasurable (fun a => ENNReal.ofReal (f a)) μ
    -/
  · exact ENNReal.continuous_ofReal.measurable.comp_aemeasurable f_int.1.aemeasurable
    /-
      🎉 no goals
    -/
    /-
      case hs
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      f_int : MeasureTheory.Integrable f μ
      f_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le 1 (f x)
      ⊢ ∀ (x : X), Membership.mem s x → LE.le 1 (ENNReal.ofReal (f x))
    -/
  · intro x hx
    /-
      case hs
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      f_int : MeasureTheory.Integrable f μ
      f_nonneg : (MeasureTheory.ae μ).EventuallyLE 0 f
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le 1 (f x)
      x : X
      hx : Membership.mem s x
      ⊢ LE.le 1 (ENNReal.ofReal (f x))
    -/
    simpa using ENNReal.ofReal_le_ofReal (hs x hx)
    /-
      🎉 no goals
    -/


lemma integral_le_measure {f : X → ℝ} {s : Set X}
    (hs : ∀ x ∈ s, f x ≤ 1) (h's : ∀ x ∈ sᶜ, f x ≤ 0) :
    ENNReal.ofReal (∫ x, f x ∂μ) ≤ μ s := by
  /-
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (μ s)
  -/
  by_cases H : Integrable f μ; swap
    /-
      case neg
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : Not (MeasureTheory.Integrable f μ)
      ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (μ s)
    -/
  · simp [integral_undef H]
    /-
      🎉 no goals
    -/
  /-
    case pos
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    H : MeasureTheory.Integrable f μ
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (μ s)
  -/
  let g x := max (f x) 0
  /-
    case pos
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    H : MeasureTheory.Integrable f μ
    g : X → Real := fun x => Max.max (f x) 0
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (μ s)
  -/
  have g_int : Integrable g μ := H.pos_part
  have : ENNReal.ofReal (∫ x, f x ∂μ) ≤ ENNReal.ofReal (∫ x, g x ∂μ) := by
    apply ENNReal.ofReal_le_ofReal
    exact integral_mono H g_int (fun x ↦ le_max_left _ _)
  /-
    case pos
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    H : MeasureTheory.Integrable f μ
    g : X → Real := fun x => Max.max (f x) 0
    g_int : MeasureTheory.Integrable g μ
    this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (μ s)
  -/
  apply this.trans
  /-
    case pos
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    H : MeasureTheory.Integrable f μ
    g : X → Real := fun x => Max.max (f x) 0
    g_int : MeasureTheory.Integrable g μ
    this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
    ⊢ LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => g x)) (μ s)
  -/
  rw [ofReal_integral_eq_lintegral_ofReal g_int (Eventually.of_forall (fun x ↦ le_max_right _ _))]
  /-
    case pos
    X : Type u_1
    inst✝ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    f : X → Real
    s : Set X
    hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
    h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
    H : MeasureTheory.Integrable f μ
    g : X → Real := fun x => Max.max (f x) 0
    g_int : MeasureTheory.Integrable g μ
    this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x => ENNReal.ofReal (g x)) (μ s)
  -/
  apply lintegral_le_meas
    /-
      case pos.hf
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : MeasureTheory.Integrable f μ
      g : X → Real := fun x => Max.max (f x) 0
      g_int : MeasureTheory.Integrable g μ
      this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
      ⊢ ∀ (a : X), LE.le (ENNReal.ofReal (g a)) 1
    -/
  · intro x
    /-
      case pos.hf
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : MeasureTheory.Integrable f μ
      g : X → Real := fun x => Max.max (f x) 0
      g_int : MeasureTheory.Integrable g μ
      this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
      x : X
      ⊢ LE.le (ENNReal.ofReal (g x)) 1
    -/
    apply ENNReal.ofReal_le_of_le_toReal
    /-
      case pos.hf.h
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : MeasureTheory.Integrable f μ
      g : X → Real := fun x => Max.max (f x) 0
      g_int : MeasureTheory.Integrable g μ
      this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
      x : X
      ⊢ LE.le (g x) (ENNReal.toReal 1)
    -/
    by_cases H : x ∈ s
      /-
        case pos
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        s : Set X
        hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
        h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
        H✝ : MeasureTheory.Integrable f μ
        g : X → Real := fun x => Max.max (f x) 0
        g_int : MeasureTheory.Integrable g μ
        this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
        x : X
        H : Membership.mem s x
        ⊢ LE.le (g x) (ENNReal.toReal 1)
      -/
    · simpa [g] using hs x H
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        s : Set X
        hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
        h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
        H✝ : MeasureTheory.Integrable f μ
        g : X → Real := fun x => Max.max (f x) 0
        g_int : MeasureTheory.Integrable g μ
        this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
        x : X
        H : Not (Membership.mem s x)
        ⊢ LE.le (g x) (ENNReal.toReal 1)
      -/
    · apply le_trans _ zero_le_one
      /-
        X : Type u_1
        inst✝ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        f : X → Real
        s : Set X
        hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
        h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
        H✝ : MeasureTheory.Integrable f μ
        g : X → Real := fun x => Max.max (f x) 0
        g_int : MeasureTheory.Integrable g μ
        this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
        x : X
        H : Not (Membership.mem s x)
        ⊢ LE.le (g x) 0
      -/
      simpa [g] using h's x H
      /-
        🎉 no goals
      -/
    /-
      case pos.h'f
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : MeasureTheory.Integrable f μ
      g : X → Real := fun x => Max.max (f x) 0
      g_int : MeasureTheory.Integrable g μ
      this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
      ⊢ ∀ (a : X), Membership.mem (HasCompl.compl s) a → Eq (ENNReal.ofReal (g a)) 0
    -/
  · intro x hx
    /-
      case pos.h'f
      X : Type u_1
      inst✝ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      f : X → Real
      s : Set X
      hs : ∀ (x : X), Membership.mem s x → LE.le (f x) 1
      h's : ∀ (x : X), Membership.mem (HasCompl.compl s) x → LE.le (f x) 0
      H : MeasureTheory.Integrable f μ
      g : X → Real := fun x => Max.max (f x) 0
      g_int : MeasureTheory.Integrable g μ
      this : LE.le (ENNReal.ofReal (MeasureTheory.integral μ fun x => f x)) (ENNReal …
      x : X
      hx : Membership.mem (HasCompl.compl s) x
      ⊢ Eq (ENNReal.ofReal (g x)) 0
    -/
    simpa [g] using h's x hx
    /-
      🎉 no goals
    -/


theorem integrableOn_iUnion_of_summable_integral_norm {f : X → E} {s : ι → Set X}
    (hs : ∀ i : ι, MeasurableSet (s i)) (hi : ∀ i : ι, IntegrableOn f (s i) μ)
    (h : Summable fun i : ι => ∫ x : X in s i, ‖f x‖ ∂μ) : IntegrableOn f (iUnion s) μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    ⊢ MeasureTheory.IntegrableOn f (Set.iUnion s) μ
  -/
  refine ⟨AEStronglyMeasurable.iUnion fun i => (hi i).1, (lintegral_iUnion_le _ _).trans_lt ?_⟩
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    ⊢ LT.lt (tsum fun i => MeasureTheory.lintegral (μ.restrict (s i)) fun a => ENo …
  -/
  have B := fun i => lintegral_coe_eq_integral (fun x : X => ‖f x‖₊) (hi i).norm
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    B : ∀ (i : ι), Eq (MeasureTheory.lintegral (μ.restrict (s i)) fun a => ↑(NNNor …
    ⊢ LT.lt (tsum fun i => MeasureTheory.lintegral (μ.restrict (s i)) fun a => ENo …
  -/
  simp_rw [enorm_eq_nnnorm, tsum_congr B]
  have S' :
    Summable fun i : ι =>
      (⟨∫ x : X in s i, ‖f x‖₊ ∂μ, setIntegral_nonneg (hs i) fun x _ => NNReal.coe_nonneg _⟩ :
        NNReal) := by
    rw [← NNReal.summable_coe]; exact h
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    B : ∀ (i : ι), Eq (MeasureTheory.lintegral (μ.restrict (s i)) fun a => ↑(NNNor …
    S' : Summable fun i => ⟨MeasureTheory.integral (μ.restrict (s i)) fun x => ↑(N …
    ⊢ LT.lt (tsum fun b => ENNReal.ofReal (MeasureTheory.integral (μ.restrict (s b …
  -/
  have S'' := ENNReal.tsum_coe_eq S'.hasSum
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    B : ∀ (i : ι), Eq (MeasureTheory.lintegral (μ.restrict (s i)) fun a => ↑(NNNor …
    S' : Summable fun i => ⟨MeasureTheory.integral (μ.restrict (s i)) fun x => ↑(N …
    S'' : Eq (tsum fun a => ↑⟨MeasureTheory.integral (μ.restrict (s a)) fun x => ↑ …
    ⊢ LT.lt (tsum fun b => ENNReal.ofReal (MeasureTheory.integral (μ.restrict (s b …
  -/
  simp_rw [ENNReal.coe_nnreal_eq, NNReal.coe_mk, coe_nnnorm] at S''
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    ι : Type u_5
    inst✝¹ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝ : NormedAddCommGroup E
    f : X → E
    s : ι → Set X
    hs : ∀ (i : ι), MeasurableSet (s i)
    hi : ∀ (i : ι), MeasureTheory.IntegrableOn f (s i) μ
    h : Summable fun i => MeasureTheory.integral (μ.restrict (s i)) fun x => Norm. …
    B : ∀ (i : ι), Eq (MeasureTheory.lintegral (μ.restrict (s i)) fun a => ↑(NNNor …
    S' : Summable fun i => ⟨MeasureTheory.integral (μ.restrict (s i)) fun x => ↑(N …
    S'' : Eq (tsum fun a => ENNReal.ofReal (MeasureTheory.integral (μ.restrict (s  …
    ⊢ LT.lt (tsum fun b => ENNReal.ofReal (MeasureTheory.integral (μ.restrict (s b …
  -/
  convert ENNReal.ofReal_lt_top
  /-
    🎉 no goals
  -/


/-- If `s` is a countable family of compact sets, `f` is a continuous function, and the sequence
`‖f.restrict (s i)‖ * μ (s i)` is summable, then `f` is integrable on the union of the `s i`. -/
theorem integrableOn_iUnion_of_summable_norm_restrict {f : C(X, E)} {s : ι → Compacts X}
    (hf : Summable fun i : ι => ‖f.restrict (s i)‖ * ENNReal.toReal (μ <| s i)) :
    IntegrableOn f (⋃ i : ι, s i) μ := by
  refine
    integrableOn_iUnion_of_summable_integral_norm (fun i => (s i).isCompact.isClosed.measurableSet)
      (fun i => (map_continuous f).continuousOn.integrableOn_compact (s i).isCompact)
      (.of_nonneg_of_le (fun ι => integral_nonneg fun x => norm_nonneg _) (fun i => ?_) hf)
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁶ : MeasurableSpace X
    ι : Type u_5
    inst✝⁵ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace X
    inst✝² : BorelSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : ContinuousMap X E
    s : ι → TopologicalSpace.Compacts X
    hf : Summable fun i => HMul.hMul (Norm.norm (ContinuousMap.restrict (↑(s i)) f …
    i : ι
    ⊢ LE.le (MeasureTheory.integral (μ.restrict ↑(s i)) fun x => Norm.norm (f x))  …
  -/
  rw [← (Real.norm_of_nonneg (integral_nonneg fun x => norm_nonneg _) : ‖_‖ = ∫ x in s i, ‖f x‖ ∂μ)]
  exact
    norm_setIntegral_le_of_norm_le_const' (s i).isCompact.measure_lt_top
      (s i).isCompact.isClosed.measurableSet fun x hx =>
      (norm_norm (f x)).symm ▸ (f.restrict (s i : Set X)).norm_coe_le_norm ⟨x, hx⟩


/-- If `s` is a countable family of compact sets covering `X`, `f` is a continuous function, and
the sequence `‖f.restrict (s i)‖ * μ (s i)` is summable, then `f` is integrable. -/
theorem integrable_of_summable_norm_restrict {f : C(X, E)} {s : ι → Compacts X}
    (hf : Summable fun i : ι => ‖f.restrict (s i)‖ * ENNReal.toReal (μ <| s i))
    (hs : ⋃ i : ι, ↑(s i) = (univ : Set X)) : Integrable f μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁶ : MeasurableSpace X
    ι : Type u_5
    inst✝⁵ : Countable ι
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : TopologicalSpace X
    inst✝² : BorelSpace X
    inst✝¹ : TopologicalSpace.MetrizableSpace X
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : ContinuousMap X E
    s : ι → TopologicalSpace.Compacts X
    hf : Summable fun i => HMul.hMul (Norm.norm (ContinuousMap.restrict (↑(s i)) f …
    hs : Eq (Set.iUnion fun i => ↑(s i)) Set.univ
    ⊢ MeasureTheory.Integrable (⇑f) μ
  -/
  simpa only [hs, integrableOn_univ] using integrableOn_iUnion_of_summable_norm_restrict hf
  /-
    🎉 no goals
  -/


/-- For `f : Lp E p μ`, we can define an element of `Lp E p (μ.restrict s)` by
`(Lp.memℒp f).restrict s).toLp f`. This map is additive. -/
theorem Lp_toLp_restrict_add (f g : Lp E p μ) (s : Set X) :
    ((Lp.memℒp (f + g)).restrict s).toLp (⇑(f + g)) =
      ((Lp.memℒp f).restrict s).toLp f + ((Lp.memℒp g).restrict s).toLp g := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    ⊢ Eq (MeasureTheory.Memℒp.toLp ↑↑(HAdd.hAdd f g) ⋯) (HAdd.hAdd (MeasureTheory. …
  -/
  ext1
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp ↑ …
  -/
  refine (ae_restrict_of_ae (Lp.coeFn_add f g)).mp ?_
  refine
    (Lp.coeFn_add (Memℒp.toLp f ((Lp.memℒp f).restrict s))
          (Memℒp.toLp g ((Lp.memℒp g).restrict s))).mp ?_
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    ⊢ Filter.Eventually (fun x => Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp ↑↑f ⋯ …
  -/
  refine (Memℒp.coeFn_toLp ((Lp.memℒp f).restrict s)).mp ?_
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    ⊢ Filter.Eventually (fun x => Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑f ⋯) x) (↑↑f x …
  -/
  refine (Memℒp.coeFn_toLp ((Lp.memℒp g).restrict s)).mp ?_
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    ⊢ Filter.Eventually (fun x => Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑g ⋯) x) (↑↑g x …
  -/
  refine (Memℒp.coeFn_toLp ((Lp.memℒp (f + g)).restrict s)).mono fun x hx1 hx2 hx3 hx4 hx5 => ?_
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    f g : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    s : Set X
    x : X
    hx1 : Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑(HAdd.hAdd f g) ⋯) x) (↑↑(HAdd.hAdd f  …
    hx2 : Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑g ⋯) x) (↑↑g x)
    hx3 : Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑f ⋯) x) (↑↑f x)
    hx4 : Eq (↑↑(HAdd.hAdd (MeasureTheory.Memℒp.toLp ↑↑f ⋯) (MeasureTheory.Memℒp.t …
    hx5 : Eq (↑↑(HAdd.hAdd f g) x) (HAdd.hAdd (↑↑f) (↑↑g) x)
    ⊢ Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑(HAdd.hAdd f g) ⋯) x) (↑↑(HAdd.hAdd (Measu …
  -/
  rw [hx4, hx1, Pi.add_apply, hx2, hx3, hx5, Pi.add_apply]
  /-
    🎉 no goals
  -/


/-- For `f : Lp E p μ`, we can define an element of `Lp E p (μ.restrict s)` by
`(Lp.memℒp f).restrict s).toLp f`. This map commutes with scalar multiplication. -/
theorem Lp_toLp_restrict_smul (c : 𝕜) (f : Lp F p μ) (s : Set X) :
    ((Lp.memℒp (c • f)).restrict s).toLp (⇑(c • f)) = c • ((Lp.memℒp f).restrict s).toLp f := by
  /-
    X : Type u_1
    F : Type u_4
    inst✝³ : MeasurableSpace X
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : ENNReal
    μ : MeasureTheory.Measure X
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    s : Set X
    ⊢ Eq (MeasureTheory.Memℒp.toLp ↑↑(HSMul.hSMul c f) ⋯) (HSMul.hSMul c (MeasureT …
  -/
  ext1
  /-
    case h
    X : Type u_1
    F : Type u_4
    inst✝³ : MeasurableSpace X
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : ENNReal
    μ : MeasureTheory.Measure X
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    s : Set X
    ⊢ (MeasureTheory.ae (μ.restrict s)).EventuallyEq ↑↑(MeasureTheory.Memℒp.toLp ↑ …
  -/
  refine (ae_restrict_of_ae (Lp.coeFn_smul c f)).mp ?_
  /-
    case h
    X : Type u_1
    F : Type u_4
    inst✝³ : MeasurableSpace X
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : ENNReal
    μ : MeasureTheory.Measure X
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    s : Set X
    ⊢ Filter.Eventually (fun x => Eq (↑↑(HSMul.hSMul c f) x) (HSMul.hSMul c (↑↑f)  …
  -/
  refine (Memℒp.coeFn_toLp ((Lp.memℒp f).restrict s)).mp ?_
  /-
    case h
    X : Type u_1
    F : Type u_4
    inst✝³ : MeasurableSpace X
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : ENNReal
    μ : MeasureTheory.Measure X
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    s : Set X
    ⊢ Filter.Eventually (fun x => Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑f ⋯) x) (↑↑f x …
  -/
  refine (Memℒp.coeFn_toLp ((Lp.memℒp (c • f)).restrict s)).mp ?_
  refine
    (Lp.coeFn_smul c (Memℒp.toLp f ((Lp.memℒp f).restrict s))).mono fun x hx1 hx2 hx3 hx4 => ?_
  /-
    case h
    X : Type u_1
    F : Type u_4
    inst✝³ : MeasurableSpace X
    𝕜 : Type u_5
    inst✝² : NormedField 𝕜
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : ENNReal
    μ : MeasureTheory.Measure X
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x
    s : Set X
    x : X
    hx1 : Eq (↑↑(HSMul.hSMul c (MeasureTheory.Memℒp.toLp ↑↑f ⋯)) x) (HSMul.hSMul c …
    hx2 : Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑(HSMul.hSMul c f) ⋯) x) (↑↑(HSMul.hSMu …
    hx3 : Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑f ⋯) x) (↑↑f x)
    hx4 : Eq (↑↑(HSMul.hSMul c f) x) (HSMul.hSMul c (↑↑f) x)
    ⊢ Eq (↑↑(MeasureTheory.Memℒp.toLp ↑↑(HSMul.hSMul c f) ⋯) x) (↑↑(HSMul.hSMul c  …
  -/
  simp only [hx2, hx1, hx3, hx4, Pi.smul_apply]
  /-
    🎉 no goals
  -/


/-- For `f : Lp E p μ`, we can define an element of `Lp E p (μ.restrict s)` by
`(Lp.memℒp f).restrict s).toLp f`. This map is non-expansive. -/
theorem norm_Lp_toLp_restrict_le (s : Set X) (f : Lp E p μ) :
    ‖((Lp.memℒp f).restrict s).toLp f‖ ≤ ‖f‖ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    s : Set X
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LE.le (Norm.norm (MeasureTheory.Memℒp.toLp ↑↑f ⋯)) (Norm.norm f)
  -/
  rw [Lp.norm_def, Lp.norm_def, eLpNorm_congr_ae (Memℒp.coeFn_toLp _)]
  /-
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    s : Set X
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p (μ.restrict s)).toReal (MeasureTheory.e …
  -/
  refine ENNReal.toReal_mono (Lp.eLpNorm_ne_top _) ?_
  /-
    X : Type u_1
    E : Type u_3
    inst✝¹ : MeasurableSpace X
    inst✝ : NormedAddCommGroup E
    p : ENNReal
    μ : MeasureTheory.Measure X
    s : Set X
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp E p μ) x
    ⊢ LE.le (MeasureTheory.eLpNorm (↑↑f) p (μ.restrict s)) (MeasureTheory.eLpNorm  …
  -/
  exact eLpNorm_mono_measure _ Measure.restrict_le_self
  /-
    🎉 no goals
  -/


variable (X F 𝕜) in
/-- Continuous linear map sending a function of `Lp F p μ` to the same function in
`Lp F p (μ.restrict s)`. -/
def LpToLpRestrictCLM (μ : Measure X) (p : ℝ≥0∞) [hp : Fact (1 ≤ p)] (s : Set X) :
    Lp F p μ →L[𝕜] Lp F p (μ.restrict s) :=
  @LinearMap.mkContinuous 𝕜 𝕜 (Lp F p μ) (Lp F p (μ.restrict s)) _ _ _ _ _ _ (RingHom.id 𝕜)
    ⟨⟨fun f => Memℒp.toLp f ((Lp.memℒp f).restrict s), fun f g => Lp_toLp_restrict_add f g s⟩,
      fun c f => Lp_toLp_restrict_smul c f s⟩
          /-
            X : Type u_1
            Y : Type u_2
            E : Type u_3
            F : Type u_4
            inst✝⁴ : MeasurableSpace X
            inst✝³ : NormedAddCommGroup E
            𝕜 : Type u_5
            inst✝² : NormedField 𝕜
            inst✝¹ : NormedAddCommGroup F
            inst✝ : NormedSpace 𝕜 F
            p✝ : ENNReal
            μ✝ μ : MeasureTheory.Measure X
            p : ENNReal
            hp : Fact (LE.le 1 p)
            s : Set X
            ⊢ ∀ (x : Subtype fun x => Membership.mem (MeasureTheory.Lp F p μ) x), LE.le (N …
          -/
    1 (by intro f; rw [one_mul]; exact norm_Lp_toLp_restrict_le s f)
                                 /-
                                   🎉 no goals
                                 -/


variable (𝕜) in
theorem LpToLpRestrictCLM_coeFn [Fact (1 ≤ p)] (s : Set X) (f : Lp F p μ) :
    LpToLpRestrictCLM X F 𝕜 μ p s f =ᵐ[μ.restrict s] f :=
  Memℒp.coeFn_toLp ((Lp.memℒp f).restrict s)


@[continuity]
theorem continuous_setIntegral [NormedSpace ℝ E] (s : Set X) :
    Continuous fun f : X →₁[μ] E => ∫ x in s, f x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : NormedSpace Real E
    s : Set X
    ⊢ Continuous fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑f x
  -/
  haveI : Fact ((1 : ℝ≥0∞) ≤ 1) := ⟨le_rfl⟩
  have h_comp :
    (fun f : X →₁[μ] E => ∫ x in s, f x ∂μ) =
      integral (μ.restrict s) ∘ fun f => LpToLpRestrictCLM X E ℝ μ 1 s f := by
    ext1 f
    rw [Function.comp_apply, integral_congr_ae (LpToLpRestrictCLM_coeFn ℝ s f)]
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : NormedSpace Real E
    s : Set X
    this : Fact (LE.le 1 1)
    h_comp : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑f x) (F …
    ⊢ Continuous fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑f x
  -/
  rw [h_comp]
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    inst✝¹ : NormedAddCommGroup E
    μ : MeasureTheory.Measure X
    inst✝ : NormedSpace Real E
    s : Set X
    this : Fact (LE.le 1 1)
    h_comp : Eq (fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑f x) (F …
    ⊢ Continuous (Function.comp (MeasureTheory.integral (μ.restrict s)) fun f => ↑ …
  -/
  exact continuous_integral.comp (LpToLpRestrictCLM X E ℝ μ 1 s).continuous
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias continuous_set_integral := continuous_setIntegral


theorem Continuous.integral_pos_of_hasCompactSupport_nonneg_nonzero [IsFiniteMeasureOnCompacts μ]
    {f : X → ℝ} {x : X} (f_cont : Continuous f) (f_comp : HasCompactSupport f) (f_nonneg : 0 ≤ f)
    (f_x : f x ≠ 0) : 0 < ∫ x, f x ∂μ :=
  integral_pos_of_integrable_nonneg_nonzero f_cont (f_cont.integrable_of_hasCompactSupport f_comp)
    f_nonneg f_x


theorem MeasureTheory.setIntegral_support : ∫ x in support F, F x ∂ν = ∫ x, F x ∂ν := by
  /-
    X : Type u_1
    M : Type u_5
    inst✝¹ : NormedAddCommGroup M
    inst✝ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    ⊢ Eq (MeasureTheory.integral (ν.restrict (Function.support F)) fun x => F x) ( …
  -/
  nth_rw 2 [← setIntegral_univ]
  /-
    X : Type u_1
    M : Type u_5
    inst✝¹ : NormedAddCommGroup M
    inst✝ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    ⊢ Eq (MeasureTheory.integral (ν.restrict (Function.support F)) fun x => F x) ( …
  -/
  rw [setIntegral_eq_of_subset_of_forall_diff_eq_zero MeasurableSet.univ (subset_univ (support F))]
  /-
    X : Type u_1
    M : Type u_5
    inst✝¹ : NormedAddCommGroup M
    inst✝ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    ⊢ ∀ (x : X), Membership.mem (SDiff.sdiff Set.univ (Function.support F)) x → Eq …
  -/
  exact fun _ hx => nmem_support.mp <| not_mem_of_mem_diff hx
  /-
    🎉 no goals
  -/


theorem MeasureTheory.setIntegral_tsupport [TopologicalSpace X] :
    ∫ x in tsupport F, F x ∂ν = ∫ x, F x ∂ν := by
  /-
    X : Type u_1
    M : Type u_5
    inst✝² : NormedAddCommGroup M
    inst✝¹ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    inst✝ : TopologicalSpace X
    ⊢ Eq (MeasureTheory.integral (ν.restrict (tsupport F)) fun x => F x) (MeasureT …
  -/
  nth_rw 2 [← setIntegral_univ]
  /-
    X : Type u_1
    M : Type u_5
    inst✝² : NormedAddCommGroup M
    inst✝¹ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    inst✝ : TopologicalSpace X
    ⊢ Eq (MeasureTheory.integral (ν.restrict (tsupport F)) fun x => F x) (MeasureT …
  -/
  rw [setIntegral_eq_of_subset_of_forall_diff_eq_zero MeasurableSet.univ (subset_univ (tsupport F))]
  /-
    X : Type u_1
    M : Type u_5
    inst✝² : NormedAddCommGroup M
    inst✝¹ : NormedSpace Real M
    mX : MeasurableSpace X
    ν : MeasureTheory.Measure X
    F : X → M
    inst✝ : TopologicalSpace X
    ⊢ ∀ (x : X), Membership.mem (SDiff.sdiff Set.univ (tsupport F)) x → Eq (F x) 0
  -/
  exact fun _ hx => image_eq_zero_of_nmem_tsupport <| not_mem_of_mem_diff hx
  /-
    🎉 no goals
  -/


/-- Fundamental theorem of calculus for set integrals:
if `μ` is a measure that is finite at a filter `l` and
`f` is a measurable function that has a finite limit `b` at `l ⊓ ae μ`, then
`∫ x in s i, f x ∂μ = μ (s i) • b + o(μ (s i))` at a filter `li` provided that
`s i` tends to `l.smallSets` along `li`.
Since `μ (s i)` is an `ℝ≥0∞` number, we use `(μ (s i)).toReal` in the actual statement.

Often there is a good formula for `(μ (s i)).toReal`, so the formalization can take an optional
argument `m` with this formula and a proof of `(fun i => (μ (s i)).toReal) =ᶠ[li] m`. Without these
arguments, `m i = (μ (s i)).toReal` is used in the output. -/
theorem Filter.Tendsto.integral_sub_linear_isLittleO_ae
    {μ : Measure X} {l : Filter X} [l.IsMeasurablyGenerated] {f : X → E} {b : E}
    (h : Tendsto f (l ⊓ ae μ) (𝓝 b)) (hfm : StronglyMeasurableAtFilter f l μ)
    (hμ : μ.FiniteAtFilter l) {s : ι → Set X} {li : Filter ι} (hs : Tendsto s li l.smallSets)
    (m : ι → ℝ := fun i => (μ (s i)).toReal)
    (hsμ : (fun i => (μ (s i)).toReal) =ᶠ[li] m := by rfl) :
    (fun i => (∫ x in s i, f x ∂μ) - m i • b) =o[li] m := by
  suffices
      (fun s => (∫ x in s, f x ∂μ) - (μ s).toReal • b) =o[l.smallSets] fun s => (μ s).toReal from
    (this.comp_tendsto hs).congr'
      (hsμ.mono fun a ha => by dsimp only [Function.comp_apply] at ha ⊢; rw [ha]) hsμ
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    ι : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure X
    l : Filter X
    inst✝ : l.IsMeasurablyGenerated
    f : X → E
    b : E
    h : Filter.Tendsto f (Min.min l (MeasureTheory.ae μ)) (nhds b)
    hfm : StronglyMeasurableAtFilter f l μ
    hμ : μ.FiniteAtFilter l
    s : ι → Set X
    li : Filter ι
    hs : Filter.Tendsto s li l.smallSets
    m : optParam (ι → Real) fun i => (μ (s i)).toReal
    hsμ : autoParam (li.EventuallyEq (fun i => (μ (s i)).toReal) m) _auto✝
    ⊢ Asymptotics.IsLittleO l.smallSets (fun s => HSub.hSub (MeasureTheory.integra …
  -/
  refine isLittleO_iff.2 fun ε ε₀ => ?_
  have : ∀ᶠ s in l.smallSets, ∀ᵐ x ∂μ, x ∈ s → f x ∈ closedBall b ε :=
    eventually_smallSets_eventually.2 (h.eventually <| closedBall_mem_nhds _ ε₀)
  filter_upwards [hμ.eventually, (hμ.integrableAtFilter_of_tendsto_ae hfm h).eventually,
    hfm.eventually, this]
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    ι : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure X
    l : Filter X
    inst✝ : l.IsMeasurablyGenerated
    f : X → E
    b : E
    h : Filter.Tendsto f (Min.min l (MeasureTheory.ae μ)) (nhds b)
    hfm : StronglyMeasurableAtFilter f l μ
    hμ : μ.FiniteAtFilter l
    s : ι → Set X
    li : Filter ι
    hs : Filter.Tendsto s li l.smallSets
    m : optParam (ι → Real) fun i => (μ (s i)).toReal
    hsμ : autoParam (li.EventuallyEq (fun i => (μ (s i)).toReal) m) _auto✝
    ε : Real
    ε₀ : LT.lt 0 ε
    this : Filter.Eventually (fun s => Filter.Eventually (fun x => Membership.mem  …
    ⊢ ∀ (a : Set X), LT.lt (μ a) Top.top → MeasureTheory.IntegrableOn f a μ → Meas …
  -/
  simp only [mem_closedBall, dist_eq_norm]
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    ι : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure X
    l : Filter X
    inst✝ : l.IsMeasurablyGenerated
    f : X → E
    b : E
    h : Filter.Tendsto f (Min.min l (MeasureTheory.ae μ)) (nhds b)
    hfm : StronglyMeasurableAtFilter f l μ
    hμ : μ.FiniteAtFilter l
    s : ι → Set X
    li : Filter ι
    hs : Filter.Tendsto s li l.smallSets
    m : optParam (ι → Real) fun i => (μ (s i)).toReal
    hsμ : autoParam (li.EventuallyEq (fun i => (μ (s i)).toReal) m) _auto✝
    ε : Real
    ε₀ : LT.lt 0 ε
    this : Filter.Eventually (fun s => Filter.Eventually (fun x => Membership.mem  …
    ⊢ ∀ (a : Set X), LT.lt (μ a) Top.top → MeasureTheory.IntegrableOn f a μ → Meas …
  -/
  intro s hμs h_integrable hfm h_norm
  rw [← setIntegral_const, ← integral_sub h_integrable (integrableOn_const.2 <| Or.inr hμs),
    Real.norm_eq_abs, abs_of_nonneg ENNReal.toReal_nonneg]
  /-
    case h
    X : Type u_1
    E : Type u_3
    inst✝⁴ : MeasurableSpace X
    ι : Type u_5
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace Real E
    inst✝¹ : CompleteSpace E
    μ : MeasureTheory.Measure X
    l : Filter X
    inst✝ : l.IsMeasurablyGenerated
    f : X → E
    b : E
    h : Filter.Tendsto f (Min.min l (MeasureTheory.ae μ)) (nhds b)
    hfm✝ : StronglyMeasurableAtFilter f l μ
    hμ : μ.FiniteAtFilter l
    s✝ : ι → Set X
    li : Filter ι
    hs : Filter.Tendsto s✝ li l.smallSets
    m : optParam (ι → Real) fun i => (μ (s✝ i)).toReal
    hsμ : autoParam (li.EventuallyEq (fun i => (μ (s✝ i)).toReal) m) _auto✝
    ε : Real
    ε₀ : LT.lt 0 ε
    this : Filter.Eventually (fun s => Filter.Eventually (fun x => Membership.mem  …
    s : Set X
    hμs : LT.lt (μ s) Top.top
    h_integrable : MeasureTheory.IntegrableOn f s μ
    hfm : MeasureTheory.AEStronglyMeasurable f (μ.restrict s)
    h_norm : Filter.Eventually (fun x => Membership.mem s x → LE.le (Norm.norm (HS …
    ⊢ LE.le (Norm.norm (MeasureTheory.integral (μ.restrict s) fun a => HSub.hSub ( …
  -/
  exact norm_setIntegral_le_of_norm_le_const_ae' hμs h_norm (hfm.sub aestronglyMeasurable_const)
  /-
    🎉 no goals
  -/


/-- Fundamental theorem of calculus for set integrals, `nhdsWithin` version: if `μ` is a locally
finite measure and `f` is an almost everywhere measurable function that is continuous at a point `a`
within a measurable set `t`, then `∫ x in s i, f x ∂μ = μ (s i) • f a + o(μ (s i))` at a filter `li`
provided that `s i` tends to `(𝓝[t] a).smallSets` along `li`.  Since `μ (s i)` is an `ℝ≥0∞`
number, we use `(μ (s i)).toReal` in the actual statement.

Often there is a good formula for `(μ (s i)).toReal`, so the formalization can take an optional
argument `m` with this formula and a proof of `(fun i => (μ (s i)).toReal) =ᶠ[li] m`. Without these
arguments, `m i = (μ (s i)).toReal` is used in the output. -/
theorem ContinuousWithinAt.integral_sub_linear_isLittleO_ae [TopologicalSpace X]
    [OpensMeasurableSpace X] {μ : Measure X}
    [IsLocallyFiniteMeasure μ] {x : X} {t : Set X} {f : X → E} (hx : ContinuousWithinAt f t x)
    (ht : MeasurableSet t) (hfm : StronglyMeasurableAtFilter f (𝓝[t] x) μ) {s : ι → Set X}
    {li : Filter ι} (hs : Tendsto s li (𝓝[t] x).smallSets) (m : ι → ℝ := fun i => (μ (s i)).toReal)
    (hsμ : (fun i => (μ (s i)).toReal) =ᶠ[li] m := by rfl) :
    (fun i => (∫ x in s i, f x ∂μ) - m i • f x) =o[li] m :=
  haveI : (𝓝[t] x).IsMeasurablyGenerated := ht.nhdsWithin_isMeasurablyGenerated _
  (hx.mono_left inf_le_left).integral_sub_linear_isLittleO_ae hfm (μ.finiteAt_nhdsWithin x t) hs m
    hsμ


/-- Fundamental theorem of calculus for set integrals, `nhds` version: if `μ` is a locally finite
measure and `f` is an almost everywhere measurable function that is continuous at a point `a`, then
`∫ x in s i, f x ∂μ = μ (s i) • f a + o(μ (s i))` at `li` provided that `s` tends to
`(𝓝 a).smallSets` along `li`. Since `μ (s i)` is an `ℝ≥0∞` number, we use `(μ (s i)).toReal` in
the actual statement.

Often there is a good formula for `(μ (s i)).toReal`, so the formalization can take an optional
argument `m` with this formula and a proof of `(fun i => (μ (s i)).toReal) =ᶠ[li] m`. Without these
arguments, `m i = (μ (s i)).toReal` is used in the output. -/
theorem ContinuousAt.integral_sub_linear_isLittleO_ae [TopologicalSpace X] [OpensMeasurableSpace X]
    {μ : Measure X} [IsLocallyFiniteMeasure μ] {x : X}
    {f : X → E} (hx : ContinuousAt f x) (hfm : StronglyMeasurableAtFilter f (𝓝 x) μ) {s : ι → Set X}
    {li : Filter ι} (hs : Tendsto s li (𝓝 x).smallSets) (m : ι → ℝ := fun i => (μ (s i)).toReal)
    (hsμ : (fun i => (μ (s i)).toReal) =ᶠ[li] m := by rfl) :
    (fun i => (∫ x in s i, f x ∂μ) - m i • f x) =o[li] m :=
  (hx.mono_left inf_le_left).integral_sub_linear_isLittleO_ae hfm (μ.finiteAt_nhds x) hs m hsμ


/-- Fundamental theorem of calculus for set integrals, `nhdsWithin` version: if `μ` is a locally
finite measure, `f` is continuous on a measurable set `t`, and `a ∈ t`, then `∫ x in (s i), f x ∂μ =
μ (s i) • f a + o(μ (s i))` at `li` provided that `s i` tends to `(𝓝[t] a).smallSets` along `li`.
Since `μ (s i)` is an `ℝ≥0∞` number, we use `(μ (s i)).toReal` in the actual statement.

Often there is a good formula for `(μ (s i)).toReal`, so the formalization can take an optional
argument `m` with this formula and a proof of `(fun i => (μ (s i)).toReal) =ᶠ[li] m`. Without these
arguments, `m i = (μ (s i)).toReal` is used in the output. -/
theorem ContinuousOn.integral_sub_linear_isLittleO_ae [TopologicalSpace X] [OpensMeasurableSpace X]
    [SecondCountableTopologyEither X E] {μ : Measure X}
    [IsLocallyFiniteMeasure μ] {x : X} {t : Set X} {f : X → E} (hft : ContinuousOn f t) (hx : x ∈ t)
    (ht : MeasurableSet t) {s : ι → Set X} {li : Filter ι} (hs : Tendsto s li (𝓝[t] x).smallSets)
    (m : ι → ℝ := fun i => (μ (s i)).toReal)
    (hsμ : (fun i => (μ (s i)).toReal) =ᶠ[li] m := by rfl) :
    (fun i => (∫ x in s i, f x ∂μ) - m i • f x) =o[li] m :=
  (hft x hx).integral_sub_linear_isLittleO_ae ht
    ⟨t, self_mem_nhdsWithin, hft.aestronglyMeasurable ht⟩ hs m hsμ


theorem integral_compLp (L : E →L[𝕜] F) (φ : Lp E p μ) :
    ∫ x, (L.compLp φ) x ∂μ = ∫ x, L (φ x) ∂μ :=
  integral_congr_ae <| coeFn_compLp _ _


theorem setIntegral_compLp (L : E →L[𝕜] F) (φ : Lp E p μ) {s : Set X} (hs : MeasurableSet s) :
    ∫ x in s, (L.compLp φ) x ∂μ = ∫ x in s, L (φ x) ∂μ :=
  setIntegral_congr_ae hs ((L.coeFn_compLp φ).mono fun _x hx _ => hx)


@[deprecated (since := "2024-04-17")]
alias set_integral_compLp := setIntegral_compLp


theorem continuous_integral_comp_L1 (L : E →L[𝕜] F) :
    Continuous fun φ : X →₁[μ] E => ∫ x : X, L (φ x) ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁶ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    inst✝ : NormedSpace Real F
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    ⊢ Continuous fun φ => MeasureTheory.integral μ fun x => L (↑↑φ x)
  -/
  rw [← funext L.integral_compLp]; exact continuous_integral.comp (L.compLpL 1 μ).continuous
                                   /-
                                     🎉 no goals
                                   -/


theorem integral_comp_comm [CompleteSpace E] (L : E →L[𝕜] F) {φ : X → E} (φ_int : Integrable φ μ) :
    ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ) := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedSpace Real F
    inst✝² : CompleteSpace F
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    φ : X → E
    φ_int : MeasureTheory.Integrable φ μ
    ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
  -/
  apply φ_int.induction (P := fun φ => ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ))
    /-
      case h_ind
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      ⊢ ∀ (c : E) ⦃s : Set X⦄, MeasurableSet s → LT.lt (μ s) Top.top → (fun φ => Eq  …
    -/
  · intro e s s_meas _
    rw [integral_indicator_const e s_meas, ← @smul_one_smul E ℝ 𝕜 _ _ _ _ _ (μ s).toReal e,
      ContinuousLinearMap.map_smul, @smul_one_smul F ℝ 𝕜 _ _ _ _ _ (μ s).toReal (L e), ←
      integral_indicator_const (L e) s_meas]
    /-
      case h_ind
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      e : E
      s : Set X
      s_meas : MeasurableSet s
      a✝ : LT.lt (μ s) Top.top
      ⊢ Eq (MeasureTheory.integral μ fun x => L (s.indicator (fun x => e) x)) (Measu …
    -/
    congr 1 with a
    /-
      case h_ind.e_f.h
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      e : E
      s : Set X
      s_meas : MeasurableSet s
      a✝ : LT.lt (μ s) Top.top
      a : X
      ⊢ Eq (L (s.indicator (fun x => e) a)) (s.indicator (fun x => L e) a)
    -/
    rw [← Function.comp_def L, Set.indicator_comp_of_zero L.map_zero, Function.comp_apply]
    /-
      🎉 no goals
    -/
    /-
      case h_add
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      ⊢ ∀ ⦃f g : X → E⦄, Disjoint (Function.support f) (Function.support g) → Measur …
    -/
  · intro f g _ f_int g_int hf hg
    simp [L.map_add, integral_add (μ := μ) f_int g_int,
      integral_add (μ := μ) (L.integrable_comp f_int) (L.integrable_comp g_int), hf, hg]
    /-
      case h_closed
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      ⊢ IsClosed (setOf fun f => (fun φ => Eq (MeasureTheory.integral μ fun x => L ( …
    -/
  · exact isClosed_eq L.continuous_integral_comp_L1 (L.continuous.comp continuous_integral)
    /-
      🎉 no goals
    -/
    /-
      case h_ae
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      ⊢ ∀ ⦃f g : X → E⦄, (MeasureTheory.ae μ).EventuallyEq f g → MeasureTheory.Integ …
    -/
  · intro f g hfg _ hf
    /-
      case h_ae
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      φ : X → E
      φ_int : MeasureTheory.Integrable φ μ
      f g : X → E
      hfg : (MeasureTheory.ae μ).EventuallyEq f g
      a✝ : MeasureTheory.Integrable f μ
      hf : Eq (MeasureTheory.integral μ fun x => L (f x)) (L (MeasureTheory.integral …
      ⊢ Eq (MeasureTheory.integral μ fun x => L (g x)) (L (MeasureTheory.integral μ  …
    -/
    convert hf using 1 <;> clear hf
      /-
        case h.e'_2
        X : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝⁹ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁸ : RCLike 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedSpace Real F
        inst✝² : CompleteSpace F
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        L : ContinuousLinearMap (RingHom.id 𝕜) E F
        φ : X → E
        φ_int : MeasureTheory.Integrable φ μ
        f g : X → E
        hfg : (MeasureTheory.ae μ).EventuallyEq f g
        a✝ : MeasureTheory.Integrable f μ
        ⊢ Eq (MeasureTheory.integral μ fun x => L (g x)) (MeasureTheory.integral μ fun …
      -/
    · exact integral_congr_ae (hfg.fun_comp L).symm
      /-
        🎉 no goals
      -/
      /-
        case h.e'_3
        X : Type u_1
        E : Type u_3
        F : Type u_4
        inst✝⁹ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁸ : RCLike 𝕜
        inst✝⁷ : NormedAddCommGroup E
        inst✝⁶ : NormedSpace 𝕜 E
        inst✝⁵ : NormedAddCommGroup F
        inst✝⁴ : NormedSpace 𝕜 F
        inst✝³ : NormedSpace Real F
        inst✝² : CompleteSpace F
        inst✝¹ : NormedSpace Real E
        inst✝ : CompleteSpace E
        L : ContinuousLinearMap (RingHom.id 𝕜) E F
        φ : X → E
        φ_int : MeasureTheory.Integrable φ μ
        f g : X → E
        hfg : (MeasureTheory.ae μ).EventuallyEq f g
        a✝ : MeasureTheory.Integrable f μ
        ⊢ Eq (L (MeasureTheory.integral μ fun x => g x)) (L (MeasureTheory.integral μ  …
      -/
    · rw [integral_congr_ae hfg.symm]
      /-
        🎉 no goals
      -/


theorem integral_apply {H : Type*} [NormedAddCommGroup H] [NormedSpace 𝕜 H] {φ : X → H →L[𝕜] E}
    (φ_int : Integrable φ μ) (v : H) : (∫ x, φ x ∂μ) v = ∫ x, φ x v ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁶ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    inst✝² : NormedSpace Real E
    H : Type u_6
    inst✝¹ : NormedAddCommGroup H
    inst✝ : NormedSpace 𝕜 H
    φ : X → ContinuousLinearMap (RingHom.id 𝕜) H E
    φ_int : MeasureTheory.Integrable φ μ
    v : H
    ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) v) (MeasureTheory.integral μ fun …
  -/
  by_cases hE : CompleteSpace E
    /-
      case pos
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedSpace Real E
      H : Type u_6
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      φ : X → ContinuousLinearMap (RingHom.id 𝕜) H E
      φ_int : MeasureTheory.Integrable φ μ
      v : H
      hE : CompleteSpace E
      ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) v) (MeasureTheory.integral μ fun …
    -/
  · exact ((ContinuousLinearMap.apply 𝕜 E v).integral_comp_comm φ_int).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁶ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      inst✝² : NormedSpace Real E
      H : Type u_6
      inst✝¹ : NormedAddCommGroup H
      inst✝ : NormedSpace 𝕜 H
      φ : X → ContinuousLinearMap (RingHom.id 𝕜) H E
      φ_int : MeasureTheory.Integrable φ μ
      v : H
      hE : Not (CompleteSpace E)
      ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) v) (MeasureTheory.integral μ fun …
    -/
  · rcases subsingleton_or_nontrivial H with hH|hH
      /-
        case neg.inl
        X : Type u_1
        E : Type u_3
        inst✝⁶ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace Real E
        H : Type u_6
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        φ : X → ContinuousLinearMap (RingHom.id 𝕜) H E
        φ_int : MeasureTheory.Integrable φ μ
        v : H
        hE : Not (CompleteSpace E)
        hH : Subsingleton H
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) v) (MeasureTheory.integral μ fun …
      -/
    · simp [Subsingleton.eq_zero v]
      /-
        🎉 no goals
      -/
    · have : ¬(CompleteSpace (H →L[𝕜] E)) := by
        rwa [SeparatingDual.completeSpace_continuousLinearMap_iff]
      /-
        case neg.inr
        X : Type u_1
        E : Type u_3
        inst✝⁶ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁵ : RCLike 𝕜
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace 𝕜 E
        inst✝² : NormedSpace Real E
        H : Type u_6
        inst✝¹ : NormedAddCommGroup H
        inst✝ : NormedSpace 𝕜 H
        φ : X → ContinuousLinearMap (RingHom.id 𝕜) H E
        φ_int : MeasureTheory.Integrable φ μ
        v : H
        hE : Not (CompleteSpace E)
        hH : Nontrivial H
        this : Not (CompleteSpace (ContinuousLinearMap (RingHom.id 𝕜) H E))
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) v) (MeasureTheory.integral μ fun …
      -/
      simp [integral, hE, this]
      /-
        🎉 no goals
      -/


theorem _root_.ContinuousMultilinearMap.integral_apply {ι : Type*} [Fintype ι] {M : ι → Type*}
    [∀ i, NormedAddCommGroup (M i)] [∀ i, NormedSpace 𝕜 (M i)]
    {φ : X → ContinuousMultilinearMap 𝕜 M E} (φ_int : Integrable φ μ) (m : ∀ i, M i) :
    (∫ x, φ x ∂μ) m = ∫ x, φ x m ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁷ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedSpace Real E
    ι : Type u_6
    inst✝² : Fintype ι
    M : ι → Type u_7
    inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
    φ : X → ContinuousMultilinearMap 𝕜 M E
    φ_int : MeasureTheory.Integrable φ μ
    m : (i : ι) → M i
    ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
  -/
  by_cases hE : CompleteSpace E
    /-
      case pos
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace Real E
      ι : Type u_6
      inst✝² : Fintype ι
      M : ι → Type u_7
      inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
      φ : X → ContinuousMultilinearMap 𝕜 M E
      φ_int : MeasureTheory.Integrable φ μ
      m : (i : ι) → M i
      hE : CompleteSpace E
      ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
    -/
  · exact ((ContinuousMultilinearMap.apply 𝕜 M E m).integral_comp_comm φ_int).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁷ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedSpace Real E
      ι : Type u_6
      inst✝² : Fintype ι
      M : ι → Type u_7
      inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
      inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
      φ : X → ContinuousMultilinearMap 𝕜 M E
      φ_int : MeasureTheory.Integrable φ μ
      m : (i : ι) → M i
      hE : Not (CompleteSpace E)
      ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
    -/
  · by_cases hm : ∀ i, m i ≠ 0
    · have : ¬ CompleteSpace (ContinuousMultilinearMap 𝕜 M E) := by
        rwa [SeparatingDual.completeSpace_continuousMultilinearMap_iff _ _ hm]
      /-
        case pos
        X : Type u_1
        E : Type u_3
        inst✝⁷ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedSpace Real E
        ι : Type u_6
        inst✝² : Fintype ι
        M : ι → Type u_7
        inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
        φ : X → ContinuousMultilinearMap 𝕜 M E
        φ_int : MeasureTheory.Integrable φ μ
        m : (i : ι) → M i
        hE : Not (CompleteSpace E)
        hm : ∀ (i : ι), Ne (m i) 0
        this : Not (CompleteSpace (ContinuousMultilinearMap 𝕜 M E))
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
      -/
      simp [integral, hE, this]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        E : Type u_3
        inst✝⁷ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedSpace Real E
        ι : Type u_6
        inst✝² : Fintype ι
        M : ι → Type u_7
        inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
        φ : X → ContinuousMultilinearMap 𝕜 M E
        φ_int : MeasureTheory.Integrable φ μ
        m : (i : ι) → M i
        hE : Not (CompleteSpace E)
        hm : Not (∀ (i : ι), Ne (m i) 0)
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
      -/
    · push_neg at hm
      /-
        case neg
        X : Type u_1
        E : Type u_3
        inst✝⁷ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedSpace Real E
        ι : Type u_6
        inst✝² : Fintype ι
        M : ι → Type u_7
        inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
        φ : X → ContinuousMultilinearMap 𝕜 M E
        φ_int : MeasureTheory.Integrable φ μ
        m : (i : ι) → M i
        hE : Not (CompleteSpace E)
        hm : Exists fun i => Eq (m i) 0
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
      -/
      rcases hm with ⟨i, hi⟩
      /-
        case neg.intro
        X : Type u_1
        E : Type u_3
        inst✝⁷ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        𝕜 : Type u_5
        inst✝⁶ : RCLike 𝕜
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace 𝕜 E
        inst✝³ : NormedSpace Real E
        ι : Type u_6
        inst✝² : Fintype ι
        M : ι → Type u_7
        inst✝¹ : (i : ι) → NormedAddCommGroup (M i)
        inst✝ : (i : ι) → NormedSpace 𝕜 (M i)
        φ : X → ContinuousMultilinearMap 𝕜 M E
        φ_int : MeasureTheory.Integrable φ μ
        m : (i : ι) → M i
        hE : Not (CompleteSpace E)
        i : ι
        hi : Eq (m i) 0
        ⊢ Eq ((MeasureTheory.integral μ fun x => φ x) m) (MeasureTheory.integral μ fun …
      -/
      simp [ContinuousMultilinearMap.map_coord_zero _ i hi]
      /-
        🎉 no goals
      -/


theorem integral_comp_comm' (L : E →L[𝕜] F) {K} (hL : AntilipschitzWith K L) (φ : X → E) :
    ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ) := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedSpace Real F
    inst✝² : CompleteSpace F
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    K : NNReal
    hL : AntilipschitzWith K ⇑L
    φ : X → E
    ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
  -/
  by_cases h : Integrable φ μ
    /-
      case pos
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁸ : RCLike 𝕜
      inst✝⁷ : NormedAddCommGroup E
      inst✝⁶ : NormedSpace 𝕜 E
      inst✝⁵ : NormedAddCommGroup F
      inst✝⁴ : NormedSpace 𝕜 F
      inst✝³ : NormedSpace Real F
      inst✝² : CompleteSpace F
      inst✝¹ : NormedSpace Real E
      inst✝ : CompleteSpace E
      L : ContinuousLinearMap (RingHom.id 𝕜) E F
      K : NNReal
      hL : AntilipschitzWith K ⇑L
      φ : X → E
      h : MeasureTheory.Integrable φ μ
      ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
    -/
  · exact integral_comp_comm L h
    /-
      🎉 no goals
    -/
  have : ¬Integrable (fun x => L (φ x)) μ := by
    rwa [← Function.comp_def,
      LipschitzWith.integrable_comp_iff_of_antilipschitz L.lipschitz hL L.map_zero]
  /-
    case neg
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁸ : RCLike 𝕜
    inst✝⁷ : NormedAddCommGroup E
    inst✝⁶ : NormedSpace 𝕜 E
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    inst✝³ : NormedSpace Real F
    inst✝² : CompleteSpace F
    inst✝¹ : NormedSpace Real E
    inst✝ : CompleteSpace E
    L : ContinuousLinearMap (RingHom.id 𝕜) E F
    K : NNReal
    hL : AntilipschitzWith K ⇑L
    φ : X → E
    h : Not (MeasureTheory.Integrable φ μ)
    this : Not (MeasureTheory.Integrable (fun x => L (φ x)) μ)
    ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
  -/
  simp [integral_undef, h, this]
  /-
    🎉 no goals
  -/


theorem integral_comp_L1_comm (L : E →L[𝕜] F) (φ : X →₁[μ] E) :
    ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ) :=
  L.integral_comp_comm (L1.integrable_coeFn φ)


theorem integral_comp_comm (L : E →ₗᵢ[𝕜] F) (φ : X → E) : ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ) :=
  L.toContinuousLinearMap.integral_comp_comm' L.antilipschitz _


theorem integral_comp_comm (L : E ≃L[𝕜] F) (φ : X → E) : ∫ x, L (φ x) ∂μ = L (∫ x, φ x ∂μ) := by
  have : CompleteSpace E ↔ CompleteSpace F :=
    completeSpace_congr (e := L.toEquiv) L.isUniformEmbedding
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁷ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝⁶ : RCLike 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedSpace Real F
    inst✝ : NormedSpace Real E
    L : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    φ : X → E
    this : Iff (CompleteSpace E) (CompleteSpace F)
    ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
  -/
  obtain ⟨_, _⟩|⟨_, _⟩ := iff_iff_and_or_not_and_not.mp this
    /-
      case inl.intro
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁷ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace Real F
      inst✝ : NormedSpace Real E
      L : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      φ : X → E
      this : Iff (CompleteSpace E) (CompleteSpace F)
      left✝ : CompleteSpace E
      right✝ : CompleteSpace F
      ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
    -/
  · exact L.toContinuousLinearMap.integral_comp_comm' L.antilipschitz _
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁷ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝⁶ : RCLike 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedSpace Real F
      inst✝ : NormedSpace Real E
      L : ContinuousLinearEquiv (RingHom.id 𝕜) E F
      φ : X → E
      this : Iff (CompleteSpace E) (CompleteSpace F)
      left✝ : Not (CompleteSpace E)
      right✝ : Not (CompleteSpace F)
      ⊢ Eq (MeasureTheory.integral μ fun x => L (φ x)) (L (MeasureTheory.integral μ  …
    -/
  · simp [integral, *]
    /-
      🎉 no goals
    -/


lemma ContinuousMap.integral_apply [NormedSpace ℝ E] [CompleteSpace E] {f : X → C(Y, E)}
    (hf : Integrable f μ) (y : Y) : (∫ x, f x ∂μ) y = ∫ x, f x y ∂μ := by
  calc (∫ x, f x ∂μ) y = ContinuousMap.evalCLM ℝ y (∫ x, f x ∂μ) := rfl
    _ = ∫ x, ContinuousMap.evalCLM ℝ y (f x) ∂μ :=
          (ContinuousLinearMap.integral_comp_comm _ hf).symm
    _ = _ := rfl


open scoped ContinuousMapZero in
theorem ContinuousMapZero.integral_apply {R : Type*} [NormedCommRing R] [Zero Y]
    [NormedAlgebra ℝ R] [CompleteSpace R] {f : X → C(Y, R)₀}
    (hf : MeasureTheory.Integrable f μ) (y : Y) :
    (∫ (x : X), f x ∂μ) y = ∫ (x : X), (f x) y ∂μ := by
  calc (∫ x, f x ∂μ) y = ContinuousMapZero.evalCLM ℝ y (∫ x, f x ∂μ) := rfl
    _ = ∫ x, ContinuousMapZero.evalCLM ℝ y (f x) ∂μ :=
          (ContinuousLinearMap.integral_comp_comm _ hf).symm
    _ = _ := rfl


@[norm_cast]
theorem integral_ofReal {f : X → ℝ} : ∫ x, (f x : 𝕜) ∂μ = ↑(∫ x, f x ∂μ) :=
  (@RCLike.ofRealLI 𝕜 _).integral_comp_comm f


theorem integral_re {f : X → 𝕜} (hf : Integrable f μ) :
    ∫ x, RCLike.re (f x) ∂μ = RCLike.re (∫ x, f x ∂μ) :=
  (@RCLike.reCLM 𝕜 _).integral_comp_comm hf


theorem integral_im {f : X → 𝕜} (hf : Integrable f μ) :
    ∫ x, RCLike.im (f x) ∂μ = RCLike.im (∫ x, f x ∂μ) :=
  (@RCLike.imCLM 𝕜 _).integral_comp_comm hf


theorem integral_conj {f : X → 𝕜} : ∫ x, conj (f x) ∂μ = conj (∫ x, f x ∂μ) :=
  (@RCLike.conjLIE 𝕜 _).toLinearIsometry.integral_comp_comm f


theorem integral_coe_re_add_coe_im {f : X → 𝕜} (hf : Integrable f μ) :
    ∫ x, (re (f x) : 𝕜) ∂μ + (∫ x, (im (f x) : 𝕜) ∂μ) * RCLike.I = ∫ x, f x ∂μ := by
  /-
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝ : RCLike 𝕜
    f : X → 𝕜
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (HAdd.hAdd (MeasureTheory.integral μ fun x => ↑(RCLike.re (f x))) (HMul.h …
  -/
  rw [mul_comm, ← smul_eq_mul, ← integral_smul, ← integral_add]
    /-
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝ : RCLike 𝕜
      f : X → 𝕜
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun a => HAdd.hAdd (↑(RCLike.re (f a))) (HSMul. …
    -/
  · congr
    /-
      case e_f
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝ : RCLike 𝕜
      f : X → 𝕜
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (fun a => HAdd.hAdd (↑(RCLike.re (f a))) (HSMul.hSMul RCLike.I ↑(RCLike.i …
    -/
    ext1 x
    /-
      case e_f.h
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝ : RCLike 𝕜
      f : X → 𝕜
      hf : MeasureTheory.Integrable f μ
      x : X
      ⊢ Eq (HAdd.hAdd (↑(RCLike.re (f x))) (HSMul.hSMul RCLike.I ↑(RCLike.im (f x))) …
    -/
    rw [smul_eq_mul, mul_comm, RCLike.re_add_im]
    /-
      🎉 no goals
    -/
    /-
      case hf
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝ : RCLike 𝕜
      f : X → 𝕜
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun x => ↑(RCLike.re (f x))) μ
    -/
  · exact hf.re.ofReal
    /-
      🎉 no goals
    -/
    /-
      case hg
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      𝕜 : Type u_5
      inst✝ : RCLike 𝕜
      f : X → 𝕜
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.Integrable (fun a => HSMul.hSMul RCLike.I ↑(RCLike.im (f a))) μ
    -/
  · exact hf.im.ofReal.smul (𝕜 := 𝕜) (β := 𝕜) RCLike.I
    /-
      🎉 no goals
    -/


theorem integral_re_add_im {f : X → 𝕜} (hf : Integrable f μ) :
    ((∫ x, RCLike.re (f x) ∂μ : ℝ) : 𝕜) + (∫ x, RCLike.im (f x) ∂μ : ℝ) * RCLike.I =
      ∫ x, f x ∂μ := by
  /-
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    𝕜 : Type u_5
    inst✝ : RCLike 𝕜
    f : X → 𝕜
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (HAdd.hAdd (↑(MeasureTheory.integral μ fun x => RCLike.re (f x))) (HMul.h …
  -/
  rw [← integral_ofReal, ← integral_ofReal, integral_coe_re_add_coe_im hf]
  /-
    🎉 no goals
  -/


theorem setIntegral_re_add_im {f : X → 𝕜} {i : Set X} (hf : IntegrableOn f i μ) :
    ((∫ x in i, RCLike.re (f x) ∂μ : ℝ) : 𝕜) + (∫ x in i, RCLike.im (f x) ∂μ : ℝ) * RCLike.I =
      ∫ x in i, f x ∂μ :=
  integral_re_add_im hf


@[deprecated (since := "2024-04-17")]
alias set_integral_re_add_im := setIntegral_re_add_im


lemma swap_integral (f : X → E × F) : (∫ x, f x ∂μ).swap = ∫ x, (f x).swap ∂μ :=
  .symm <| (ContinuousLinearEquiv.prodComm ℝ E F).integral_comp_comm f


theorem fst_integral [CompleteSpace F] {f : X → E × F} (hf : Integrable f μ) :
    (∫ x, f x ∂μ).1 = ∫ x, (f x).1 ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace F
    f : X → Prod E F
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x).1 (MeasureTheory.integral μ fun x …
  -/
  by_cases hE : CompleteSpace E
    /-
      case pos
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      f : X → Prod E F
      hf : MeasureTheory.Integrable f μ
      hE : CompleteSpace E
      ⊢ Eq (MeasureTheory.integral μ fun x => f x).1 (MeasureTheory.integral μ fun x …
    -/
  · exact ((ContinuousLinearMap.fst ℝ E F).integral_comp_comm hf).symm
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      f : X → Prod E F
      hf : MeasureTheory.Integrable f μ
      hE : Not (CompleteSpace E)
      ⊢ Eq (MeasureTheory.integral μ fun x => f x).1 (MeasureTheory.integral μ fun x …
    -/
  · have : ¬(CompleteSpace (E × F)) := fun h ↦ hE <| .fst_of_prod (β := F)
    /-
      case neg
      X : Type u_1
      E : Type u_3
      F : Type u_4
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace Real E
      inst✝¹ : NormedSpace Real F
      inst✝ : CompleteSpace F
      f : X → Prod E F
      hf : MeasureTheory.Integrable f μ
      hE : Not (CompleteSpace E)
      this : Not (CompleteSpace (Prod E F))
      ⊢ Eq (MeasureTheory.integral μ fun x => f x).1 (MeasureTheory.integral μ fun x …
    -/
    simp [integral, *]
    /-
      🎉 no goals
    -/


theorem snd_integral [CompleteSpace E] {f : X → E × F} (hf : Integrable f μ) :
    (∫ x, f x ∂μ).2 = ∫ x, (f x).2 ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace E
    f : X → Prod E F
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => f x).2 (MeasureTheory.integral μ fun x …
  -/
  rw [← Prod.fst_swap, swap_integral]
  /-
    X : Type u_1
    E : Type u_3
    F : Type u_4
    inst✝⁵ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace Real E
    inst✝¹ : NormedSpace Real F
    inst✝ : CompleteSpace E
    f : X → Prod E F
    hf : MeasureTheory.Integrable f μ
    ⊢ Eq (MeasureTheory.integral μ fun x => (f x).swap).1 (MeasureTheory.integral  …
  -/
  exact fst_integral <| hf.snd.prod_mk hf.fst
  /-
    🎉 no goals
  -/


theorem integral_pair [CompleteSpace E] [CompleteSpace F] {f : X → E} {g : X → F}
    (hf : Integrable f μ) (hg : Integrable g μ) :
    ∫ x, (f x, g x) ∂μ = (∫ x, f x ∂μ, ∫ x, g x ∂μ) :=
  have := hf.prod_mk hg
  Prod.ext (fst_integral this) (snd_integral this)


theorem integral_smul_const {𝕜 : Type*} [RCLike 𝕜] [NormedSpace 𝕜 E] [CompleteSpace E]
    (f : X → 𝕜) (c : E) :
    ∫ x, f x • c ∂μ = (∫ x, f x ∂μ) • c := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝⁵ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    𝕜 : Type u_6
    inst✝² : RCLike 𝕜
    inst✝¹ : NormedSpace 𝕜 E
    inst✝ : CompleteSpace E
    f : X → 𝕜
    c : E
    ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) c) (HSMul.hSMul (Mea …
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_6
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : X → 𝕜
      c : E
      hf : MeasureTheory.Integrable f μ
      ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) c) (HSMul.hSMul (Mea …
    -/
  · exact ((1 : 𝕜 →L[𝕜] 𝕜).smulRight c).integral_comp_comm hf
    /-
      🎉 no goals
    -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_6
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : X → 𝕜
      c : E
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) c) (HSMul.hSMul (Mea …
    -/
  · by_cases hc : c = 0
      /-
        case pos
        X : Type u_1
        E : Type u_3
        inst✝⁵ : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝⁴ : NormedAddCommGroup E
        inst✝³ : NormedSpace Real E
        𝕜 : Type u_6
        inst✝² : RCLike 𝕜
        inst✝¹ : NormedSpace 𝕜 E
        inst✝ : CompleteSpace E
        f : X → 𝕜
        c : E
        hf : Not (MeasureTheory.Integrable f μ)
        hc : Eq c 0
        ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) c) (HSMul.hSMul (Mea …
      -/
    · simp [hc, integral_zero, smul_zero]
      /-
        🎉 no goals
      -/
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_6
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : X → 𝕜
      c : E
      hf : Not (MeasureTheory.Integrable f μ)
      hc : Not (Eq c 0)
      ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) c) (HSMul.hSMul (Mea …
    -/
    rw [integral_undef hf, integral_undef, zero_smul]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_6
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : X → 𝕜
      c : E
      hf : Not (MeasureTheory.Integrable f μ)
      hc : Not (Eq c 0)
      ⊢ Not (MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) c) μ)
    -/
    rw [integrable_smul_const hc]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝⁵ : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      𝕜 : Type u_6
      inst✝² : RCLike 𝕜
      inst✝¹ : NormedSpace 𝕜 E
      inst✝ : CompleteSpace E
      f : X → 𝕜
      c : E
      hf : Not (MeasureTheory.Integrable f μ)
      hc : Not (Eq c 0)
      ⊢ Not (MeasureTheory.Integrable f μ)
    -/
    simp_rw [hf, not_false_eq_true]
    /-
      🎉 no goals
    -/


theorem integral_withDensity_eq_integral_smul {f : X → ℝ≥0} (f_meas : Measurable f) (g : X → E) :
    ∫ x, g x ∂μ.withDensity (fun x => f x) = ∫ x, f x • g x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → NNReal
    f_meas : Measurable f
    g : X → E
    ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => g x) (Me …
  -/
  by_cases hE : CompleteSpace E; swap; · simp [integral, hE]
                                         /-
                                           🎉 no goals
                                         -/
  /-
    case pos
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → NNReal
    f_meas : Measurable f
    g : X → E
    hE : CompleteSpace E
    ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => g x) (Me …
  -/
  by_cases hg : Integrable g (μ.withDensity fun x => f x); swap
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : Not (MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x)))
      ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => g x) (Me …
    -/
  · rw [integral_undef hg, integral_undef]
    /-
      case neg
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : Not (MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x)))
      ⊢ Not (MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (g x)) μ)
    -/
    rwa [← integrable_withDensity_iff_integrable_smul f_meas]
    /-
      🎉 no goals
    -/
  refine Integrable.induction
    (P := fun g => ∫ x, g x ∂μ.withDensity (fun x => f x) = ∫ x, f x • g x ∂μ) ?_ ?_ ?_ ?_ hg
    /-
      case pos.refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      ⊢ ∀ (c : E) ⦃s : Set X⦄, MeasurableSet s → LT.lt ((μ.withDensity fun x => ↑(f  …
    -/
  · intro c s s_meas hs
    /-
      case pos.refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      c : E
      s : Set X
      s_meas : MeasurableSet s
      hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => s.indica …
    -/
    rw [integral_indicator s_meas]
    /-
      case pos.refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      c : E
      s : Set X
      s_meas : MeasurableSet s
      hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
      ⊢ Eq (MeasureTheory.integral ((μ.withDensity fun x => ↑(f x)).restrict s) fun  …
    -/
    simp_rw [← indicator_smul_apply, integral_indicator s_meas]
    /-
      case pos.refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      c : E
      s : Set X
      s_meas : MeasurableSet s
      hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
      ⊢ Eq (MeasureTheory.integral ((μ.withDensity fun x => ↑(f x)).restrict s) fun  …
    -/
    simp only [s_meas, integral_const, Measure.restrict_apply', univ_inter, withDensity_apply]
    /-
      case pos.refine_1
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      c : E
      s : Set X
      s_meas : MeasurableSet s
      hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
      ⊢ Eq (HSMul.hSMul (MeasureTheory.lintegral (μ.restrict s) fun x => ↑(f x)).toR …
    -/
    rw [lintegral_coe_eq_integral, ENNReal.toReal_ofReal, ← integral_smul_const]
      /-
        case pos.refine_1
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
        ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HSMul.hSMul (↑(f x)) c) ( …
      -/
    · rfl
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_1
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
        ⊢ LE.le 0 (MeasureTheory.integral (μ.restrict s) fun a => ↑(f a))
      -/
    · exact integral_nonneg fun x => NNReal.coe_nonneg _
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_1.hfi
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
        ⊢ MeasureTheory.Integrable (fun x => ↑(f x)) (μ.restrict s)
      -/
    · refine ⟨f_meas.coe_nnreal_real.aemeasurable.aestronglyMeasurable, ?_⟩
      /-
        case pos.refine_1.hfi
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt ((μ.withDensity fun x => ↑(f x)) s) Top.top
        ⊢ MeasureTheory.HasFiniteIntegral (fun x => ↑(f x)) (μ.restrict s)
      -/
      rw [withDensity_apply _ s_meas] at hs
      /-
        case pos.refine_1.hfi
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt (MeasureTheory.lintegral (μ.restrict s) fun a => ↑(f a)) Top.top
        ⊢ MeasureTheory.HasFiniteIntegral (fun x => ↑(f x)) (μ.restrict s)
      -/
      rw [hasFiniteIntegral_iff_nnnorm]
      /-
        case pos.refine_1.hfi
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt (MeasureTheory.lintegral (μ.restrict s) fun a => ↑(f a)) Top.top
        ⊢ LT.lt (MeasureTheory.lintegral (μ.restrict s) fun a => ↑(NNNorm.nnnorm ↑(f a …
      -/
      convert hs with x
      /-
        case h.e'_3.h.e'_4.h.h.e'_1
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        c : E
        s : Set X
        s_meas : MeasurableSet s
        hs : LT.lt (MeasureTheory.lintegral (μ.restrict s) fun a => ↑(f a)) Top.top
        x : X
        ⊢ Eq (NNNorm.nnnorm ↑(f x)) (f x)
      -/
      simp only [NNReal.nnnorm_eq]
      /-
        🎉 no goals
      -/
    /-
      case pos.refine_2
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      ⊢ ∀ ⦃f_1 g : X → E⦄, Disjoint (Function.support f_1) (Function.support g) → Me …
    -/
  · intro u u' _ u_int u'_int h h'
    change
      (∫ x : X, u x + u' x ∂μ.withDensity fun x : X => ↑(f x)) = ∫ x : X, f x • (u x + u' x) ∂μ
    /-
      case pos.refine_2
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u u' : X → E
      a✝ : Disjoint (Function.support u) (Function.support u')
      u_int : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      u'_int : MeasureTheory.Integrable u' (μ.withDensity fun x => ↑(f x))
      h : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x) ( …
      h' : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u' x) …
      ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => HAdd.hAd …
    -/
    simp_rw [smul_add]
    /-
      case pos.refine_2
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u u' : X → E
      a✝ : Disjoint (Function.support u) (Function.support u')
      u_int : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      u'_int : MeasureTheory.Integrable u' (μ.withDensity fun x => ↑(f x))
      h : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x) ( …
      h' : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u' x) …
      ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => HAdd.hAd …
    -/
    rw [integral_add u_int u'_int, h, h', integral_add]
      /-
        case pos.refine_2.hf
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        u u' : X → E
        a✝ : Disjoint (Function.support u) (Function.support u')
        u_int : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
        u'_int : MeasureTheory.Integrable u' (μ.withDensity fun x => ↑(f x))
        h : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x) ( …
        h' : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u' x) …
        ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (u x)) μ
      -/
    · exact (integrable_withDensity_iff_integrable_smul f_meas).1 u_int
      /-
        🎉 no goals
      -/
      /-
        case pos.refine_2.hg
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        u u' : X → E
        a✝ : Disjoint (Function.support u) (Function.support u')
        u_int : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
        u'_int : MeasureTheory.Integrable u' (μ.withDensity fun x => ↑(f x))
        h : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x) ( …
        h' : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u' x) …
        ⊢ MeasureTheory.Integrable (fun x => HSMul.hSMul (f x) (u' x)) μ
      -/
    · exact (integrable_withDensity_iff_integrable_smul f_meas).1 u'_int
      /-
        🎉 no goals
      -/
  · have C1 :
      Continuous fun u : Lp E 1 (μ.withDensity fun x => f x) =>
        ∫ x, u x ∂μ.withDensity fun x => f x :=
      continuous_integral
    have C2 : Continuous fun u : Lp E 1 (μ.withDensity fun x => f x) => ∫ x, f x • u x ∂μ := by
      have : Continuous ((fun u : Lp E 1 μ => ∫ x, u x ∂μ) ∘ withDensitySMulLI (E := E) μ f_meas) :=
        continuous_integral.comp (withDensitySMulLI (E := E) μ f_meas).continuous
      convert this with u
      simp only [Function.comp_apply, withDensitySMulLI_apply]
      exact integral_congr_ae (memℒ1_smul_of_L1_withDensity f_meas u).coeFn_toLp.symm
    /-
      case pos.refine_3
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      C1 : Continuous fun u => MeasureTheory.integral (μ.withDensity fun x => ↑(f x) …
      C2 : Continuous fun u => MeasureTheory.integral μ fun x => HSMul.hSMul (f x) ( …
      ⊢ IsClosed (setOf fun f_1 => (fun g => Eq (MeasureTheory.integral (μ.withDensi …
    -/
    exact isClosed_eq C1 C2
    /-
      🎉 no goals
    -/
    /-
      case pos.refine_4
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      ⊢ ∀ ⦃f_1 g : X → E⦄, (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).Eventu …
    -/
  · intro u v huv _ hu
    /-
      case pos.refine_4
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u v : X → E
      huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
      a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
      ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => v x) (Me …
    -/
    rw [← integral_congr_ae huv, hu]
    /-
      case pos.refine_4
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u v : X → E
      huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
      a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
      ⊢ Eq (MeasureTheory.integral μ fun x => HSMul.hSMul (f x) (u x)) (MeasureTheor …
    -/
    apply integral_congr_ae
    /-
      case pos.refine_4.h
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u v : X → E
      huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
      a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
      ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (f a) (u a)) fun a = …
    -/
    filter_upwards [(ae_withDensity_iff f_meas.coe_nnreal_ennreal).1 huv] with x hx
    /-
      case h
      X : Type u_1
      E : Type u_3
      inst✝² : MeasurableSpace X
      μ : MeasureTheory.Measure X
      inst✝¹ : NormedAddCommGroup E
      inst✝ : NormedSpace Real E
      f : X → NNReal
      f_meas : Measurable f
      g : X → E
      hE : CompleteSpace E
      hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
      u v : X → E
      huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
      a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
      hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
      x : X
      hx : Ne (↑(f x)) 0 → Eq (u x) (v x)
      ⊢ Eq (HSMul.hSMul (f x) (u x)) (HSMul.hSMul (f x) (v x))
    -/
    rcases eq_or_ne (f x) 0 with (h'x | h'x)
      /-
        case h.inl
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        u v : X → E
        huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
        a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
        hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
        x : X
        hx : Ne (↑(f x)) 0 → Eq (u x) (v x)
        h'x : Eq (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (u x)) (HSMul.hSMul (f x) (v x))
      -/
    · simp only [h'x, zero_smul]
      /-
        🎉 no goals
      -/
      /-
        case h.inr
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        u v : X → E
        huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
        a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
        hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
        x : X
        hx : Ne (↑(f x)) 0 → Eq (u x) (v x)
        h'x : Ne (f x) 0
        ⊢ Eq (HSMul.hSMul (f x) (u x)) (HSMul.hSMul (f x) (v x))
      -/
    · rw [hx _]
      /-
        X : Type u_1
        E : Type u_3
        inst✝² : MeasurableSpace X
        μ : MeasureTheory.Measure X
        inst✝¹ : NormedAddCommGroup E
        inst✝ : NormedSpace Real E
        f : X → NNReal
        f_meas : Measurable f
        g : X → E
        hE : CompleteSpace E
        hg : MeasureTheory.Integrable g (μ.withDensity fun x => ↑(f x))
        u v : X → E
        huv : (MeasureTheory.ae (μ.withDensity fun x => ↑(f x))).EventuallyEq u v
        a✝ : MeasureTheory.Integrable u (μ.withDensity fun x => ↑(f x))
        hu : Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => u x)  …
        x : X
        hx : Ne (↑(f x)) 0 → Eq (u x) (v x)
        h'x : Ne (f x) 0
        ⊢ Ne (↑(f x)) 0
      -/
      simpa only [Ne, ENNReal.coe_eq_zero] using h'x
      /-
        🎉 no goals
      -/


theorem integral_withDensity_eq_integral_smul₀ {f : X → ℝ≥0} (hf : AEMeasurable f μ) (g : X → E) :
    ∫ x, g x ∂μ.withDensity (fun x => f x) = ∫ x, f x • g x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → NNReal
    hf : AEMeasurable f μ
    g : X → E
    ⊢ Eq (MeasureTheory.integral (μ.withDensity fun x => ↑(f x)) fun x => g x) (Me …
  -/
  let f' := hf.mk _
  calc
    ∫ x, g x ∂μ.withDensity (fun x => f x) = ∫ x, g x ∂μ.withDensity fun x => f' x := by
      congr 1
      apply withDensity_congr_ae
      filter_upwards [hf.ae_eq_mk] with x hx
      rw [hx]
    _ = ∫ x, f' x • g x ∂μ := integral_withDensity_eq_integral_smul hf.measurable_mk _
    _ = ∫ x, f x • g x ∂μ := by
      apply integral_congr_ae
      filter_upwards [hf.ae_eq_mk] with x hx
      rw [hx]


theorem setIntegral_withDensity_eq_setIntegral_smul {f : X → ℝ≥0} (f_meas : Measurable f)
    (g : X → E) {s : Set X} (hs : MeasurableSet s) :
    ∫ x in s, g x ∂μ.withDensity (fun x => f x) = ∫ x in s, f x • g x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → NNReal
    f_meas : Measurable f
    g : X → E
    s : Set X
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((μ.withDensity fun x => ↑(f x)).restrict s) fun  …
  -/
  rw [restrict_withDensity hs, integral_withDensity_eq_integral_smul f_meas]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_withDensity_eq_set_integral_smul := setIntegral_withDensity_eq_setIntegral_smul


theorem setIntegral_withDensity_eq_setIntegral_smul₀ {f : X → ℝ≥0} {s : Set X}
    (hf : AEMeasurable f (μ.restrict s)) (g : X → E) (hs : MeasurableSet s) :
    ∫ x in s, g x ∂μ.withDensity (fun x => f x) = ∫ x in s, f x • g x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝² : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝¹ : NormedAddCommGroup E
    inst✝ : NormedSpace Real E
    f : X → NNReal
    s : Set X
    hf : AEMeasurable f (μ.restrict s)
    g : X → E
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral ((μ.withDensity fun x => ↑(f x)).restrict s) fun  …
  -/
  rw [restrict_withDensity hs, integral_withDensity_eq_integral_smul₀ hf]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_withDensity_eq_set_integral_smul₀ := setIntegral_withDensity_eq_setIntegral_smul₀


theorem setIntegral_withDensity_eq_setIntegral_smul₀' [SFinite μ] {f : X → ℝ≥0} (s : Set X)
    (hf : AEMeasurable f (μ.restrict s)) (g : X → E)  :
    ∫ x in s, g x ∂μ.withDensity (fun x => f x) = ∫ x in s, f x • g x ∂μ := by
  /-
    X : Type u_1
    E : Type u_3
    inst✝³ : MeasurableSpace X
    μ : MeasureTheory.Measure X
    inst✝² : NormedAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : MeasureTheory.SFinite μ
    f : X → NNReal
    s : Set X
    hf : AEMeasurable f (μ.restrict s)
    g : X → E
    ⊢ Eq (MeasureTheory.integral ((μ.withDensity fun x => ↑(f x)).restrict s) fun  …
  -/
  rw [restrict_withDensity' s, integral_withDensity_eq_integral_smul₀ hf]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_withDensity_eq_set_integral_smul₀' :=
  setIntegral_withDensity_eq_setIntegral_smul₀'


theorem measure_le_lintegral_thickenedIndicatorAux (μ : Measure X) {E : Set X}
    (E_mble : MeasurableSet E) (δ : ℝ) : μ E ≤ ∫⁻ x, (thickenedIndicatorAux δ E x : ℝ≥0∞) ∂μ := by
  /-
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    inst✝ : PseudoEMetricSpace X
    μ : MeasureTheory.Measure X
    E : Set X
    E_mble : MeasurableSet E
    δ : Real
    ⊢ LE.le (μ E) (MeasureTheory.lintegral μ fun x => thickenedIndicatorAux δ E x)
  -/
  convert_to lintegral μ (E.indicator fun _ => (1 : ℝ≥0∞)) ≤ lintegral μ (thickenedIndicatorAux δ E)
    /-
      case h.e'_3
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      inst✝ : PseudoEMetricSpace X
      μ : MeasureTheory.Measure X
      E : Set X
      E_mble : MeasurableSet E
      δ : Real
      ⊢ Eq (μ E) (MeasureTheory.lintegral μ (E.indicator fun x => 1))
    -/
  · rw [lintegral_indicator E_mble]
    /-
      case h.e'_3
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      inst✝ : PseudoEMetricSpace X
      μ : MeasureTheory.Measure X
      E : Set X
      E_mble : MeasurableSet E
      δ : Real
      ⊢ Eq (μ E) (MeasureTheory.lintegral (μ.restrict E) fun a => 1)
    -/
    simp only [lintegral_one, Measure.restrict_apply, MeasurableSet.univ, univ_inter]
    /-
      🎉 no goals
    -/
    /-
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      inst✝ : PseudoEMetricSpace X
      μ : MeasureTheory.Measure X
      E : Set X
      E_mble : MeasurableSet E
      δ : Real
      ⊢ LE.le (MeasureTheory.lintegral μ (E.indicator fun x => 1)) (MeasureTheory.li …
    -/
  · apply lintegral_mono
    /-
      case hfg
      X : Type u_1
      inst✝¹ : MeasurableSpace X
      inst✝ : PseudoEMetricSpace X
      μ : MeasureTheory.Measure X
      E : Set X
      E_mble : MeasurableSet E
      δ : Real
      ⊢ LE.le (E.indicator fun x => 1) (thickenedIndicatorAux δ E)
    -/
    apply indicator_le_thickenedIndicatorAux
    /-
      🎉 no goals
    -/


theorem measure_le_lintegral_thickenedIndicator (μ : Measure X) {E : Set X}
    (E_mble : MeasurableSet E) {δ : ℝ} (δ_pos : 0 < δ) :
    μ E ≤ ∫⁻ x, (thickenedIndicator δ_pos E x : ℝ≥0∞) ∂μ := by
  /-
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    inst✝ : PseudoEMetricSpace X
    μ : MeasureTheory.Measure X
    E : Set X
    E_mble : MeasurableSet E
    δ : Real
    δ_pos : LT.lt 0 δ
    ⊢ LE.le (μ E) (MeasureTheory.lintegral μ fun x => ↑((thickenedIndicator δ_pos  …
  -/
  convert measure_le_lintegral_thickenedIndicatorAux μ E_mble δ
  /-
    case h.e'_4.h.e'_4.h
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    inst✝ : PseudoEMetricSpace X
    μ : MeasureTheory.Measure X
    E : Set X
    E_mble : MeasurableSet E
    δ : Real
    δ_pos : LT.lt 0 δ
    x✝ : X
    ⊢ Eq (↑((thickenedIndicator δ_pos E) x✝)) (thickenedIndicatorAux δ E x✝)
  -/
  dsimp
  /-
    case h.e'_4.h.e'_4.h
    X : Type u_1
    inst✝¹ : MeasurableSpace X
    inst✝ : PseudoEMetricSpace X
    μ : MeasureTheory.Measure X
    E : Set X
    E_mble : MeasurableSet E
    δ : Real
    δ_pos : LT.lt 0 δ
    x✝ : X
    ⊢ Eq (↑(thickenedIndicatorAux δ E x✝).toNNReal) (thickenedIndicatorAux δ E x✝)
  -/
  simp only [thickenedIndicatorAux_lt_top.ne, ENNReal.coe_toNNReal, Ne, not_false_iff]
  /-
    🎉 no goals
  -/


theorem Integrable.simpleFunc_mul (g : SimpleFunc X ℝ) (hf : Integrable f μ) :
    Integrable (⇑g * f) μ := by
  refine
    SimpleFunc.induction (fun c s hs => ?_)
      (fun g₁ g₂ _ h_int₁ h_int₂ =>
        (h_int₁.add h_int₂).congr (by rw [SimpleFunc.coe_add, add_mul]))
      g
  simp only [SimpleFunc.const_zero, SimpleFunc.coe_piecewise, SimpleFunc.coe_const,
    SimpleFunc.coe_zero, Set.piecewise_eq_indicator]
  have : Set.indicator s (Function.const X c) * f = s.indicator (c • f) := by
    ext1 x
    by_cases hx : x ∈ s
    · simp only [hx, Pi.mul_apply, Set.indicator_of_mem, Pi.smul_apply, Algebra.id.smul_eq_mul,
        ← Function.const_def]
    · simp only [hx, Pi.mul_apply, Set.indicator_of_not_mem, not_false_iff, zero_mul]
  /-
    X : Type u_6
    f : X → Real
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    g : MeasureTheory.SimpleFunc X Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    s : Set X
    hs : MeasurableSet s
    this : Eq (HMul.hMul (s.indicator (Function.const X c)) f) (s.indicator (HSMul …
    ⊢ MeasureTheory.Integrable (HMul.hMul (s.indicator (Function.const X c)) f) μ
  -/
  rw [this, integrable_indicator_iff hs]
  /-
    X : Type u_6
    f : X → Real
    m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    g : MeasureTheory.SimpleFunc X Real
    hf : MeasureTheory.Integrable f μ
    c : Real
    s : Set X
    hs : MeasurableSet s
    this : Eq (HMul.hMul (s.indicator (Function.const X c)) f) (s.indicator (HSMul …
    ⊢ MeasureTheory.IntegrableOn (HSMul.hSMul c f) s μ
  -/
  exact (hf.smul c).integrableOn
  /-
    🎉 no goals
  -/


theorem Integrable.simpleFunc_mul' (hm : m ≤ m0) (g : @SimpleFunc X m ℝ) (hf : Integrable f μ) :
    Integrable (⇑g * f) μ := by
  /-
    X : Type u_6
    f : X → Real
    m m0 : MeasurableSpace X
    μ : MeasureTheory.Measure X
    hm : LE.le m m0
    g : MeasureTheory.SimpleFunc X Real
    hf : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.Integrable (HMul.hMul (⇑g) f) μ
  -/
  rw [← SimpleFunc.coe_toLargerSpace_eq hm g]; exact hf.simpleFunc_mul (g.toLargerSpace hm)
                                               /-
                                                 🎉 no goals
                                               -/


/-- The parametric integral over a continuous function on a compact set is continuous,
  under mild assumptions on the topologies involved. -/
theorem continuous_parametric_integral_of_continuous
    [FirstCountableTopology X] [LocallyCompactSpace X]
    [SecondCountableTopologyEither Y E] [IsLocallyFiniteMeasure μ]
    {f : X → Y → E} (hf : Continuous f.uncurry) {s : Set Y} (hs : IsCompact s) :
    Continuous (∫ y in s, f · y ∂μ) := by
  /-
    Y : Type u_2
    E : Type u_3
    X : Type u_5
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace Y
    inst✝⁶ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FirstCountableTopology X
    inst✝² : LocallyCompactSpace X
    inst✝¹ : SecondCountableTopologyEither Y E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    s : Set Y
    hs : IsCompact s
    ⊢ Continuous fun x => MeasureTheory.integral (μ.restrict s) fun y => f x y
  -/
  rw [continuous_iff_continuousAt]
  /-
    Y : Type u_2
    E : Type u_3
    X : Type u_5
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace Y
    inst✝⁶ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FirstCountableTopology X
    inst✝² : LocallyCompactSpace X
    inst✝¹ : SecondCountableTopologyEither Y E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    s : Set Y
    hs : IsCompact s
    ⊢ ∀ (x : X), ContinuousAt (fun x => MeasureTheory.integral (μ.restrict s) fun  …
  -/
  intro x₀
  /-
    Y : Type u_2
    E : Type u_3
    X : Type u_5
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace Y
    inst✝⁶ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FirstCountableTopology X
    inst✝² : LocallyCompactSpace X
    inst✝¹ : SecondCountableTopologyEither Y E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    s : Set Y
    hs : IsCompact s
    x₀ : X
    ⊢ ContinuousAt (fun x => MeasureTheory.integral (μ.restrict s) fun y => f x y) …
  -/
  rcases exists_compact_mem_nhds x₀ with ⟨U, U_cpct, U_nhds⟩
  /-
    case intro.intro
    Y : Type u_2
    E : Type u_3
    X : Type u_5
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace Y
    inst✝⁶ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FirstCountableTopology X
    inst✝² : LocallyCompactSpace X
    inst✝¹ : SecondCountableTopologyEither Y E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    s : Set Y
    hs : IsCompact s
    x₀ : X
    U : Set X
    U_cpct : IsCompact U
    U_nhds : Membership.mem (nhds x₀) U
    ⊢ ContinuousAt (fun x => MeasureTheory.integral (μ.restrict s) fun y => f x y) …
  -/
  rcases (U_cpct.prod hs).bddAbove_image hf.norm.continuousOn with ⟨M, hM⟩
  /-
    case intro.intro.intro
    Y : Type u_2
    E : Type u_3
    X : Type u_5
    inst✝⁹ : TopologicalSpace X
    inst✝⁸ : TopologicalSpace Y
    inst✝⁷ : MeasurableSpace Y
    inst✝⁶ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace Real E
    inst✝³ : FirstCountableTopology X
    inst✝² : LocallyCompactSpace X
    inst✝¹ : SecondCountableTopologyEither Y E
    inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
    f : X → Y → E
    hf : Continuous (Function.uncurry f)
    s : Set Y
    hs : IsCompact s
    x₀ : X
    U : Set X
    U_cpct : IsCompact U
    U_nhds : Membership.mem (nhds x₀) U
    M : Real
    hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
    ⊢ ContinuousAt (fun x => MeasureTheory.integral (μ.restrict s) fun y => f x y) …
  -/
  apply continuousAt_of_dominated
    /-
      case intro.intro.intro.hF_meas
      Y : Type u_2
      E : Type u_3
      X : Type u_5
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : TopologicalSpace Y
      inst✝⁷ : MeasurableSpace Y
      inst✝⁶ : OpensMeasurableSpace Y
      μ : MeasureTheory.Measure Y
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FirstCountableTopology X
      inst✝² : LocallyCompactSpace X
      inst✝¹ : SecondCountableTopologyEither Y E
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      f : X → Y → E
      hf : Continuous (Function.uncurry f)
      s : Set Y
      hs : IsCompact s
      x₀ : X
      U : Set X
      U_cpct : IsCompact U
      U_nhds : Membership.mem (nhds x₀) U
      M : Real
      hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
      ⊢ Filter.Eventually (fun x => MeasureTheory.AEStronglyMeasurable (f x) (μ.rest …
    -/
  · filter_upwards with x using Continuous.aestronglyMeasurable (by fun_prop)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.h_bound
      Y : Type u_2
      E : Type u_3
      X : Type u_5
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : TopologicalSpace Y
      inst✝⁷ : MeasurableSpace Y
      inst✝⁶ : OpensMeasurableSpace Y
      μ : MeasureTheory.Measure Y
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FirstCountableTopology X
      inst✝² : LocallyCompactSpace X
      inst✝¹ : SecondCountableTopologyEither Y E
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      f : X → Y → E
      hf : Continuous (Function.uncurry f)
      s : Set Y
      hs : IsCompact s
      x₀ : X
      U : Set X
      U_cpct : IsCompact U
      U_nhds : Membership.mem (nhds x₀) U
      M : Real
      hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
      ⊢ Filter.Eventually (fun x => Filter.Eventually (fun a => LE.le (Norm.norm (f  …
    -/
  · filter_upwards [U_nhds] with x x_in
    /-
      case h
      Y : Type u_2
      E : Type u_3
      X : Type u_5
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : TopologicalSpace Y
      inst✝⁷ : MeasurableSpace Y
      inst✝⁶ : OpensMeasurableSpace Y
      μ : MeasureTheory.Measure Y
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FirstCountableTopology X
      inst✝² : LocallyCompactSpace X
      inst✝¹ : SecondCountableTopologyEither Y E
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      f : X → Y → E
      hf : Continuous (Function.uncurry f)
      s : Set Y
      hs : IsCompact s
      x₀ : X
      U : Set X
      U_cpct : IsCompact U
      U_nhds : Membership.mem (nhds x₀) U
      M : Real
      hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
      x : X
      x_in : Membership.mem U x
      ⊢ Filter.Eventually (fun a => LE.le (Norm.norm (f x a)) (?intro.intro.intro.bo …
    -/
    rw [ae_restrict_iff]
      /-
        case h
        Y : Type u_2
        E : Type u_3
        X : Type u_5
        inst✝⁹ : TopologicalSpace X
        inst✝⁸ : TopologicalSpace Y
        inst✝⁷ : MeasurableSpace Y
        inst✝⁶ : OpensMeasurableSpace Y
        μ : MeasureTheory.Measure Y
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : FirstCountableTopology X
        inst✝² : LocallyCompactSpace X
        inst✝¹ : SecondCountableTopologyEither Y E
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        f : X → Y → E
        hf : Continuous (Function.uncurry f)
        s : Set Y
        hs : IsCompact s
        x₀ : X
        U : Set X
        U_cpct : IsCompact U
        U_nhds : Membership.mem (nhds x₀) U
        M : Real
        hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
        x : X
        x_in : Membership.mem U x
        ⊢ Filter.Eventually (fun x_1 => Membership.mem s x_1 → LE.le (Norm.norm (f x x …
      -/
    · filter_upwards with t t_in using hM (mem_image_of_mem _ <| mk_mem_prod x_in t_in)
      /-
        🎉 no goals
      -/
      /-
        case h
        Y : Type u_2
        E : Type u_3
        X : Type u_5
        inst✝⁹ : TopologicalSpace X
        inst✝⁸ : TopologicalSpace Y
        inst✝⁷ : MeasurableSpace Y
        inst✝⁶ : OpensMeasurableSpace Y
        μ : MeasureTheory.Measure Y
        inst✝⁵ : NormedAddCommGroup E
        inst✝⁴ : NormedSpace Real E
        inst✝³ : FirstCountableTopology X
        inst✝² : LocallyCompactSpace X
        inst✝¹ : SecondCountableTopologyEither Y E
        inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
        f : X → Y → E
        hf : Continuous (Function.uncurry f)
        s : Set Y
        hs : IsCompact s
        x₀ : X
        U : Set X
        U_cpct : IsCompact U
        U_nhds : Membership.mem (nhds x₀) U
        M : Real
        hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
        x : X
        x_in : Membership.mem U x
        ⊢ MeasurableSet (setOf fun x_1 => LE.le (Norm.norm (f x x_1)) M)
      -/
    · exact (isClosed_le (by fun_prop) (by fun_prop)).measurableSet
      /-
        🎉 no goals
      -/
    /-
      case intro.intro.intro.bound_integrable
      Y : Type u_2
      E : Type u_3
      X : Type u_5
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : TopologicalSpace Y
      inst✝⁷ : MeasurableSpace Y
      inst✝⁶ : OpensMeasurableSpace Y
      μ : MeasureTheory.Measure Y
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FirstCountableTopology X
      inst✝² : LocallyCompactSpace X
      inst✝¹ : SecondCountableTopologyEither Y E
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      f : X → Y → E
      hf : Continuous (Function.uncurry f)
      s : Set Y
      hs : IsCompact s
      x₀ : X
      U : Set X
      U_cpct : IsCompact U
      U_nhds : Membership.mem (nhds x₀) U
      M : Real
      hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
      ⊢ MeasureTheory.Integrable (fun t => M) (μ.restrict s)
    -/
  · exact integrableOn_const.mpr (Or.inr hs.measure_lt_top)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.h_cont
      Y : Type u_2
      E : Type u_3
      X : Type u_5
      inst✝⁹ : TopologicalSpace X
      inst✝⁸ : TopologicalSpace Y
      inst✝⁷ : MeasurableSpace Y
      inst✝⁶ : OpensMeasurableSpace Y
      μ : MeasureTheory.Measure Y
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace Real E
      inst✝³ : FirstCountableTopology X
      inst✝² : LocallyCompactSpace X
      inst✝¹ : SecondCountableTopologyEither Y E
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      f : X → Y → E
      hf : Continuous (Function.uncurry f)
      s : Set Y
      hs : IsCompact s
      x₀ : X
      U : Set X
      U_cpct : IsCompact U
      U_nhds : Membership.mem (nhds x₀) U
      M : Real
      hM : Membership.mem (upperBounds (Set.image (fun x => Norm.norm (Function.uncu …
      ⊢ Filter.Eventually (fun a => ContinuousAt (fun x => f x a) x₀) (MeasureTheory …
    -/
  · filter_upwards using (by fun_prop)
    /-
      🎉 no goals
    -/


/-- Consider a parameterized integral `x ↦ ∫ y, L (g y) (f x y)` where `L` is bilinear,
`g` is locally integrable and `f` is continuous and uniformly compactly supported. Then the
integral depends continuously on `x`. -/
lemma continuousOn_integral_bilinear_of_locally_integrable_of_compact_support
    [NormedSpace 𝕜 E] (L : F →L[𝕜] G →L[𝕜] E)
    {f : X → Y → G} {s : Set X} {k : Set Y} {g : Y → F}
    (hk : IsCompact k) (hf : ContinuousOn f.uncurry (s ×ˢ univ))
    (hfs : ∀ p, ∀ x, p ∈ s → x ∉ k → f p x = 0) (hg : IntegrableOn g k μ) :
    ContinuousOn (fun x ↦ ∫ y, L (g y) (f x y) ∂μ) s := by
  have A : ∀ p ∈ s, Continuous (f p) := fun p hp ↦ by
    refine hf.comp_continuous (continuous_const.prod_mk continuous_id') fun y => ?_
    simpa only [prod_mk_mem_set_prod_eq, mem_univ, and_true] using hp
  /-
    Y : Type u_2
    E : Type u_3
    F : Type u_4
    X : Type u_5
    G : Type u_6
    𝕜 : Type u_7
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : TopologicalSpace Y
    inst✝⁹ : MeasurableSpace Y
    inst✝⁸ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : NormedSpace 𝕜 E
    L : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜) G …
    f : X → Y → G
    s : Set X
    k : Set Y
    g : Y → F
    hk : IsCompact k
    hf : ContinuousOn (Function.uncurry f) (SProd.sprod s Set.univ)
    hfs : ∀ (p : X) (x : Y), Membership.mem s p → Not (Membership.mem k x) → Eq (f …
    hg : MeasureTheory.IntegrableOn g k μ
    A : ∀ (p : X), Membership.mem s p → Continuous (f p)
    ⊢ ContinuousOn (fun x => MeasureTheory.integral μ fun y => (L (g y)) (f x y)) s
  -/
  intro q hq
  /-
    Y : Type u_2
    E : Type u_3
    F : Type u_4
    X : Type u_5
    G : Type u_6
    𝕜 : Type u_7
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : TopologicalSpace Y
    inst✝⁹ : MeasurableSpace Y
    inst✝⁸ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : NormedSpace 𝕜 E
    L : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜) G …
    f : X → Y → G
    s : Set X
    k : Set Y
    g : Y → F
    hk : IsCompact k
    hf : ContinuousOn (Function.uncurry f) (SProd.sprod s Set.univ)
    hfs : ∀ (p : X) (x : Y), Membership.mem s p → Not (Membership.mem k x) → Eq (f …
    hg : MeasureTheory.IntegrableOn g k μ
    A : ∀ (p : X), Membership.mem s p → Continuous (f p)
    q : X
    hq : Membership.mem s q
    ⊢ ContinuousWithinAt (fun x => MeasureTheory.integral μ fun y => (L (g y)) (f  …
  -/
  apply Metric.continuousWithinAt_iff'.2 (fun ε εpos ↦ ?_)
  obtain ⟨δ, δpos, hδ⟩ : ∃ (δ : ℝ), 0 < δ ∧ ∫ x in k, ‖L‖ * ‖g x‖ * δ ∂μ < ε := by
    simpa [integral_mul_right] using exists_pos_mul_lt εpos _
  obtain ⟨v, v_mem, hv⟩ : ∃ v ∈ 𝓝[s] q, ∀ p ∈ v, ∀ x ∈ k, dist (f p x) (f q x) < δ :=
    hk.mem_uniformity_of_prod
      (hf.mono (Set.prod_mono_right (subset_univ k))) hq (dist_mem_uniformity δpos)
  /-
    case intro.intro.intro.intro
    Y : Type u_2
    E : Type u_3
    F : Type u_4
    X : Type u_5
    G : Type u_6
    𝕜 : Type u_7
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : TopologicalSpace Y
    inst✝⁹ : MeasurableSpace Y
    inst✝⁸ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : NormedSpace 𝕜 E
    L : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜) G …
    f : X → Y → G
    s : Set X
    k : Set Y
    g : Y → F
    hk : IsCompact k
    hf : ContinuousOn (Function.uncurry f) (SProd.sprod s Set.univ)
    hfs : ∀ (p : X) (x : Y), Membership.mem s p → Not (Membership.mem k x) → Eq (f …
    hg : MeasureTheory.IntegrableOn g k μ
    A : ∀ (p : X), Membership.mem s p → Continuous (f p)
    q : X
    hq : Membership.mem s q
    ε : Real
    εpos : GT.gt ε 0
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt (MeasureTheory.integral (μ.restrict k) fun x => HMul.hMul (HMul.hMu …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q s) v
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Y), Membership.mem k x → LT.lt (Di …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (MeasureTheory.integral μ fun y …
  -/
  simp_rw [dist_eq_norm] at hv ⊢
  have I : ∀ p ∈ s, IntegrableOn (fun y ↦ L (g y) (f p y)) k μ := by
    intro p hp
    obtain ⟨C, hC⟩ : ∃ C, ∀ y, ‖f p y‖ ≤ C := by
      have : ContinuousOn (f p) k := by
        have : ContinuousOn (fun y ↦ (p, y)) k := by fun_prop
        exact hf.comp this (by simp [MapsTo, hp])
      rcases IsCompact.exists_bound_of_continuousOn hk this with ⟨C, hC⟩
      refine ⟨max C 0, fun y ↦ ?_⟩
      by_cases hx : y ∈ k
      · exact (hC y hx).trans (le_max_left _ _)
      · simp [hfs p y hp hx]
    have : IntegrableOn (fun y ↦ ‖L‖ * ‖g y‖ * C) k μ :=
      (hg.norm.const_mul _).mul_const _
    apply Integrable.mono' this ?_ ?_
    · borelize G
      apply L.aestronglyMeasurable_comp₂ hg.aestronglyMeasurable
      apply StronglyMeasurable.aestronglyMeasurable
      apply Continuous.stronglyMeasurable_of_support_subset_isCompact (A p hp) hk
      apply support_subset_iff'.2 (fun y hy ↦ hfs p y hp hy)
    · apply Eventually.of_forall (fun y ↦ (le_opNorm₂ L (g y) (f p y)).trans ?_)
      gcongr
      apply hC
  /-
    case intro.intro.intro.intro
    Y : Type u_2
    E : Type u_3
    F : Type u_4
    X : Type u_5
    G : Type u_6
    𝕜 : Type u_7
    inst✝¹¹ : TopologicalSpace X
    inst✝¹⁰ : TopologicalSpace Y
    inst✝⁹ : MeasurableSpace Y
    inst✝⁸ : OpensMeasurableSpace Y
    μ : MeasureTheory.Measure Y
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup E
    inst✝⁵ : NormedSpace Real E
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    inst✝² : NormedAddCommGroup G
    inst✝¹ : NormedSpace 𝕜 G
    inst✝ : NormedSpace 𝕜 E
    L : ContinuousLinearMap (RingHom.id 𝕜) F (ContinuousLinearMap (RingHom.id 𝕜) G …
    f : X → Y → G
    s : Set X
    k : Set Y
    g : Y → F
    hk : IsCompact k
    hf : ContinuousOn (Function.uncurry f) (SProd.sprod s Set.univ)
    hfs : ∀ (p : X) (x : Y), Membership.mem s p → Not (Membership.mem k x) → Eq (f …
    hg : MeasureTheory.IntegrableOn g k μ
    A : ∀ (p : X), Membership.mem s p → Continuous (f p)
    q : X
    hq : Membership.mem s q
    ε : Real
    εpos : GT.gt ε 0
    δ : Real
    δpos : LT.lt 0 δ
    hδ : LT.lt (MeasureTheory.integral (μ.restrict k) fun x => HMul.hMul (HMul.hMu …
    v : Set X
    v_mem : Membership.mem (nhdsWithin q s) v
    hv : ∀ (p : X), Membership.mem v p → ∀ (x : Y), Membership.mem k x → LT.lt (No …
    I : ∀ (p : X), Membership.mem s p → MeasureTheory.IntegrableOn (fun y => (L (g …
    ⊢ Filter.Eventually (fun x => LT.lt (Norm.norm (HSub.hSub (MeasureTheory.integ …
  -/
  filter_upwards [v_mem, self_mem_nhdsWithin] with p hp h'p
  calc
  ‖∫ x, L (g x) (f p x) ∂μ - ∫ x, L (g x) (f q x) ∂μ‖
    = ‖∫ x in k, L (g x) (f p x) ∂μ - ∫ x in k, L (g x) (f q x) ∂μ‖ := by
      congr 2
      · refine (setIntegral_eq_integral_of_forall_compl_eq_zero (fun y hy ↦ ?_)).symm
        simp [hfs p y h'p hy]
      · refine (setIntegral_eq_integral_of_forall_compl_eq_zero (fun y hy ↦ ?_)).symm
        simp [hfs q y hq hy]
  _ = ‖∫ x in k, L (g x) (f p x) - L (g x) (f q x) ∂μ‖ := by rw [integral_sub (I p h'p) (I q hq)]
  _ ≤ ∫ x in k, ‖L (g x) (f p x) - L (g x) (f q x)‖ ∂μ := norm_integral_le_integral_norm _
  _ ≤ ∫ x in k, ‖L‖ * ‖g x‖ * δ ∂μ := by
      apply integral_mono_of_nonneg (Eventually.of_forall (fun y ↦ by positivity))
      · exact (hg.norm.const_mul _).mul_const _
      · filter_upwards with y
        by_cases hy : y ∈ k
        · dsimp only
          specialize hv p hp y hy
          calc
          ‖L (g y) (f p y) - L (g y) (f q y)‖
            = ‖L (g y) (f p y - f q y)‖ := by simp only [map_sub]
          _ ≤ ‖L‖ * ‖g y‖ * ‖f p y - f q y‖ := le_opNorm₂ _ _ _
          _ ≤ ‖L‖ * ‖g y‖ * δ := by gcongr
        · simp only [hfs p y h'p hy, hfs q y hq hy, sub_self, norm_zero, mul_zero]
          positivity
  _ < ε := hδ


/-- Consider a parameterized integral `x ↦ ∫ y, f x y` where `f` is continuous and uniformly
compactly supported. Then the integral depends continuously on `x`. -/
lemma continuousOn_integral_of_compact_support
    {f : X → Y → E} {s : Set X} {k : Set Y} [IsFiniteMeasureOnCompacts μ]
    (hk : IsCompact k) (hf : ContinuousOn f.uncurry (s ×ˢ univ))
    (hfs : ∀ p, ∀ x, p ∈ s → x ∉ k → f p x = 0) :
    ContinuousOn (fun x ↦ ∫ y, f x y ∂μ) s := by
  simpa using continuousOn_integral_bilinear_of_locally_integrable_of_compact_support (lsmul ℝ ℝ)
    hk hf hfs (integrableOn_const.2 (Or.inr hk.measure_lt_top)) (μ := μ) (g := fun _ ↦ 1)


