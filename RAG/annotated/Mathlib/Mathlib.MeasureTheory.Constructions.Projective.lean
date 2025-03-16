/-- A family of measures indexed by finite sets of `ι` is projective if, for finite sets `J ⊆ I`,
the projection from `∀ i : I, α i` to `∀ i : J, α i` maps `P I` to `P J`. -/
def IsProjectiveMeasureFamily (P : ∀ J : Finset ι, Measure (∀ j : J, α j)) : Prop :=
  ∀ (I J : Finset ι) (hJI : J ⊆ I),
    P J = (P I).map (Finset.restrict₂ hJI)


lemma eq_zero_of_isEmpty [h : IsEmpty (Π i, α i)]
    (hP : IsProjectiveMeasureFamily P) (I : Finset ι) :
    P I = 0 := by
  classical
  obtain ⟨i, hi⟩ := isEmpty_pi.mp h
  rw [hP (insert i I) I (I.subset_insert i)]
  have : IsEmpty (Π j : ↑(insert i I), α j) := by simp [hi]
  rw [(P (insert i I)).eq_zero_of_isEmpty]
  simp


/-- Auxiliary lemma for `measure_univ_eq`. -/
lemma measure_univ_eq_of_subset (hP : IsProjectiveMeasureFamily P) (hJI : J ⊆ I) :
    P I univ = P J univ := by
  classical
  have : (univ : Set (∀ i : I, α i)) =
      Finset.restrict₂ hJI ⁻¹' (univ : Set (∀ i : J, α i)) := by
    rw [preimage_univ]
  rw [this, ← Measure.map_apply _ MeasurableSet.univ]
  · rw [hP I J hJI]
  · exact measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _)


lemma measure_univ_eq (hP : IsProjectiveMeasureFamily P) (I J : Finset ι) :
    P I univ = P J univ := by
  classical
  rw [← hP.measure_univ_eq_of_subset I.subset_union_left,
    ← hP.measure_univ_eq_of_subset (I.subset_union_right (s₂ := J))]


lemma congr_cylinder_of_subset (hP : IsProjectiveMeasureFamily P)
    {S : Set (∀ i : I, α i)} {T : Set (∀ i : J, α i)} (hT : MeasurableSet T)
    (h_eq : cylinder I S = cylinder J T) (hJI : J ⊆ I) :
    P I S = P J T := by
  cases isEmpty_or_nonempty (∀ i, α i) with
  | inl h =>
    suffices ∀ I, P I univ = 0 by
      simp only [Measure.measure_univ_eq_zero] at this
      simp [this]
    intro I
    simp only [isEmpty_pi] at h
    obtain ⟨i, hi_empty⟩ := h
    rw [measure_univ_eq hP I {i}]
    have : (univ : Set ((j : {x // x ∈ ({i} : Finset ι)}) → α j)) = ∅ := by simp [hi_empty]
    simp [this]
  | inr h =>
    have : S = Finset.restrict₂ hJI ⁻¹' T :=
      eq_of_cylinder_eq_of_subset h_eq hJI
    rw [hP I J hJI, Measure.map_apply _ hT, this]
    exact measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _)


lemma congr_cylinder (hP : IsProjectiveMeasureFamily P)
    {S : Set (∀ i : I, α i)} {T : Set (∀ i : J, α i)} (hS : MeasurableSet S) (hT : MeasurableSet T)
    (h_eq : cylinder I S = cylinder J T) :
    P I S = P J T := by
  classical
  let U := Finset.restrict₂ Finset.subset_union_left ⁻¹' S ∩
      Finset.restrict₂ Finset.subset_union_right ⁻¹' T
  suffices P (I ∪ J) U = P I S ∧ P (I ∪ J) U = P J T from this.1.symm.trans this.2
  constructor
  · have h_eq_union : cylinder I S = cylinder (I ∪ J) U := by
      rw [← inter_cylinder, h_eq, inter_self]
    exact hP.congr_cylinder_of_subset hS h_eq_union.symm Finset.subset_union_left
  · have h_eq_union : cylinder J T = cylinder (I ∪ J) U := by
      rw [← inter_cylinder, h_eq, inter_self]
    exact hP.congr_cylinder_of_subset hT h_eq_union.symm Finset.subset_union_right


/-- A measure `μ` is the projective limit of a family of measures indexed by finite sets of `ι` if
for all `I : Finset ι`, the projection from `∀ i, α i` to `∀ i : I, α i` maps `μ` to `P I`. -/
def IsProjectiveLimit (μ : Measure (∀ i, α i))
    (P : ∀ J : Finset ι, Measure (∀ j : J, α j)) : Prop :=
  ∀ I : Finset ι, (μ.map I.restrict) = P I


lemma measure_cylinder (h : IsProjectiveLimit μ P)
    (I : Finset ι) {s : Set (∀ i : I, α i)} (hs : MeasurableSet s) :
    μ (cylinder I s) = P I s := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    h : MeasureTheory.IsProjectiveLimit μ P
    I : Finset ι
    s : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    hs : MeasurableSet s
    ⊢ Eq (μ (MeasureTheory.cylinder I s)) ((P I) s)
  -/
  rw [cylinder, ← Measure.map_apply _ hs, h I]
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    h : MeasureTheory.IsProjectiveLimit μ P
    I : Finset ι
    s : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    hs : MeasurableSet s
    ⊢ Measurable I.restrict
  -/
  exact measurable_pi_lambda _ (fun _ ↦ measurable_pi_apply _)
  /-
    🎉 no goals
  -/


lemma measure_univ_eq (hμ : IsProjectiveLimit μ P) (I : Finset ι) :
    μ univ = P I univ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    I : Finset ι
    ⊢ Eq (μ Set.univ) ((P I) Set.univ)
  -/
  rw [← cylinder_univ I, hμ.measure_cylinder _ MeasurableSet.univ]
  /-
    🎉 no goals
  -/


lemma isFiniteMeasure [∀ i, IsFiniteMeasure (P i)] (hμ : IsProjectiveLimit μ P) :
    IsFiniteMeasure μ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ MeasureTheory.IsFiniteMeasure μ
  -/
  constructor
  /-
    case measure_univ_lt_top
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ LT.lt (μ Set.univ) Top.top
  -/
  rw [hμ.measure_univ_eq (∅ : Finset ι)]
  /-
    case measure_univ_lt_top
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ LT.lt ((P EmptyCollection.emptyCollection) Set.univ) Top.top
  -/
  exact measure_lt_top _ _
  /-
    🎉 no goals
  -/


lemma isProbabilityMeasure [∀ i, IsProbabilityMeasure (P i)] (hμ : IsProjectiveLimit μ P) :
    IsProbabilityMeasure μ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsProbabilityMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ MeasureTheory.IsProbabilityMeasure μ
  -/
  constructor
  /-
    case measure_univ
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsProbabilityMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ Eq (μ Set.univ) 1
  -/
  rw [hμ.measure_univ_eq (∅ : Finset ι)]
  /-
    case measure_univ
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsProbabilityMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    ⊢ Eq ((P EmptyCollection.emptyCollection) Set.univ) 1
  -/
  exact measure_univ
  /-
    🎉 no goals
  -/


lemma measure_univ_unique (hμ : IsProjectiveLimit μ P) (hν : IsProjectiveLimit ν P) :
    μ univ = ν univ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ ν : MeasureTheory.Measure ((i : ι) → α i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    hν : MeasureTheory.IsProjectiveLimit ν P
    ⊢ Eq (μ Set.univ) (ν Set.univ)
  -/
  rw [hμ.measure_univ_eq (∅ : Finset ι), hν.measure_univ_eq (∅ : Finset ι)]
  /-
    🎉 no goals
  -/


/-- The projective limit of a family of finite measures is unique. -/
theorem unique [∀ i, IsFiniteMeasure (P i)]
    (hμ : IsProjectiveLimit μ P) (hν : IsProjectiveLimit ν P) :
    μ = ν := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ ν : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    hν : MeasureTheory.IsProjectiveLimit ν P
    ⊢ Eq μ ν
  -/
  haveI : IsFiniteMeasure μ := hμ.isFiniteMeasure
  refine ext_of_generate_finite (measurableCylinders α) generateFrom_measurableCylinders.symm
    isPiSystem_measurableCylinders (fun s hs ↦ ?_) (hμ.measure_univ_unique hν)
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ ν : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    hν : MeasureTheory.IsProjectiveLimit ν P
    this : MeasureTheory.IsFiniteMeasure μ
    s : Set ((i : ι) → α i)
    hs : Membership.mem (MeasureTheory.measurableCylinders α) s
    ⊢ Eq (μ s) (ν s)
  -/
  obtain ⟨I, S, hS, rfl⟩ := (mem_measurableCylinders _).mp hs
  /-
    case intro.intro.intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝¹ : (i : ι) → MeasurableSpace (α i)
    P : (J : Finset ι) → MeasureTheory.Measure ((j : Subtype fun x => Membership.m …
    μ ν : MeasureTheory.Measure ((i : ι) → α i)
    inst✝ : ∀ (i : Finset ι), MeasureTheory.IsFiniteMeasure (P i)
    hμ : MeasureTheory.IsProjectiveLimit μ P
    hν : MeasureTheory.IsProjectiveLimit ν P
    this : MeasureTheory.IsFiniteMeasure μ
    I : Finset ι
    S : Set ((i : Subtype fun x => Membership.mem I x) → α ↑i)
    hS : MeasurableSet S
    hs : Membership.mem (MeasureTheory.measurableCylinders α) (MeasureTheory.cylin …
    ⊢ Eq (μ (MeasureTheory.cylinder I S)) (ν (MeasureTheory.cylinder I S))
  -/
  rw [hμ.measure_cylinder _ hS, hν.measure_cylinder _ hS]
  /-
    🎉 no goals
  -/


