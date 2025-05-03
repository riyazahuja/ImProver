/-- A `BoxIntegral.Box` has integral vertices if its vertices have coordinates in `ℤ`. -/
def BoxIntegral.hasIntegralVertices (B : Box ι) : Prop :=
  ∃ l u : ι → ℤ, (∀ i, B.lower i = l i) ∧ (∀ i, B.upper i = u i)


/-- Any bounded set is contained in a `BoxIntegral.Box` with integral vertices. -/
theorem BoxIntegral.le_hasIntegralVertices_of_isBounded [Finite ι] {s : Set (ι → ℝ)}
    (h : IsBounded s) :
    ∃ B : BoxIntegral.Box ι, hasIntegralVertices B ∧ s ≤ B := by
  /-
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    ⊢ Exists fun B => And (BoxIntegral.hasIntegralVertices B) (LE.le s ↑B)
  -/
  have := Fintype.ofFinite ι
  /-
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    this : Fintype ι
    ⊢ Exists fun B => And (BoxIntegral.hasIntegralVertices B) (LE.le s ↑B)
  -/
  obtain ⟨R, hR₁, hR₂⟩ := IsBounded.subset_ball_lt h 0 0
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    this : Fintype ι
    R : Real
    hR₁ : LT.lt 0 R
    hR₂ : HasSubset.Subset s (Metric.ball 0 R)
    ⊢ Exists fun B => And (BoxIntegral.hasIntegralVertices B) (LE.le s ↑B)
  -/
  let C : ℕ := ⌈R⌉₊
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    this : Fintype ι
    R : Real
    hR₁ : LT.lt 0 R
    hR₂ : HasSubset.Subset s (Metric.ball 0 R)
    C : Nat := Nat.ceil R
    ⊢ Exists fun B => And (BoxIntegral.hasIntegralVertices B) (LE.le s ↑B)
  -/
  have hC := Nat.ceil_pos.mpr hR₁
  let I : Box ι := Box.mk (fun _ ↦ - C) (fun _ ↦ C )
    (fun _ ↦ by simp [C, neg_lt_self_iff, Nat.cast_pos, hC])
  refine ⟨I, ⟨fun _ ↦ - C, fun _ ↦ C, fun i ↦ (Int.cast_neg_natCast C).symm, fun _ ↦ rfl⟩,
    le_trans hR₂ ?_⟩
  suffices Metric.ball (0 : ι → ℝ) C ≤ I from
    le_trans (Metric.ball_subset_ball (Nat.le_ceil R)) this
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    this : Fintype ι
    R : Real
    hR₁ : LT.lt 0 R
    hR₂ : HasSubset.Subset s (Metric.ball 0 R)
    C : Nat := Nat.ceil R
    hC : LT.lt 0 (Nat.ceil R)
    I : BoxIntegral.Box ι := { lower := fun x => Neg.neg ↑C, upper := fun x => ↑C, …
    ⊢ LE.le (Metric.ball 0 ↑C) ↑I
  -/
  intro x hx
  simp_rw [C, mem_ball_zero_iff, pi_norm_lt_iff (Nat.cast_pos.mpr hC),
    Real.norm_eq_abs, abs_lt] at hx
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : Finite ι
    s : Set (ι → Real)
    h : Bornology.IsBounded s
    this : Fintype ι
    R : Real
    hR₁ : LT.lt 0 R
    hR₂ : HasSubset.Subset s (Metric.ball 0 R)
    C : Nat := Nat.ceil R
    hC : LT.lt 0 (Nat.ceil R)
    I : BoxIntegral.Box ι := { lower := fun x => Neg.neg ↑C, upper := fun x => ↑C, …
    x : ι → Real
    hx : ∀ (i : ι), And (LT.lt (Neg.neg ↑(Nat.ceil R)) (x i)) (LT.lt (x i) ↑(Nat.c …
    ⊢ Membership.mem (↑I) x
  -/
  exact fun i ↦ ⟨(hx i).1, le_of_lt (hx i).2⟩
  /-
    🎉 no goals
  -/


/-- A `BoxIntegral`, indexed by a positive integer `n` and `ν : ι → ℤ`, with corners `ν i / n`
and of side length `1 / n`. -/
def box [NeZero n] (ν : ι → ℤ) : Box ι where
  lower := fun i ↦ ν i / n
  upper := fun i ↦ (ν i + 1) / n
                               /-
                                 ι : Type u_1
                                 n : Nat
                                 inst✝ : NeZero n
                                 ν : ι → Int
                                 x✝ : ι
                                 ⊢ LT.lt ((fun i => HDiv.hDiv ↑(ν i) ↑n) x✝) ((fun i => HDiv.hDiv (HAdd.hAdd (↑ …
                               -/
  lower_lt_upper := fun _ ↦ by norm_num [add_div, n.pos_of_neZero]
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem box_lower [NeZero n] (ν : ι → ℤ) :
    (box n ν).lower = fun i ↦ (ν i / n : ℝ) := rfl


@[simp]
theorem box_upper [NeZero n] (ν : ι → ℤ) :
    (box n ν).upper = fun i ↦ ((ν i + 1)/ n : ℝ) := rfl


variable {n} in
@[simp]
theorem mem_box_iff [NeZero n] {ν : ι → ℤ} {x : ι → ℝ} :
    x ∈ box n ν ↔ ∀ i, ν i / n < x i ∧ x i ≤ (ν i + 1) / n := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x : ι → Real
    ⊢ Iff (Membership.mem (BoxIntegral.unitPartition.box n ν) x) (∀ (i : ι), And ( …
  -/
  simp_rw [Box.mem_def, box, Set.mem_Ioc]
  /-
    🎉 no goals
  -/


variable {n} in
theorem mem_box_iff' [NeZero n] {ν : ι → ℤ} {x : ι → ℝ} :
    x ∈ box n ν ↔ ∀ i, ν i < n * x i ∧ n * x i ≤ ν i + 1 := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x : ι → Real
    ⊢ Iff (Membership.mem (BoxIntegral.unitPartition.box n ν) x) (∀ (i : ι), And ( …
  -/
  have h : 0 < (n : ℝ) := Nat.cast_pos.mpr <| n.pos_of_neZero
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x : ι → Real
    h : LT.lt 0 ↑n
    ⊢ Iff (Membership.mem (BoxIntegral.unitPartition.box n ν) x) (∀ (i : ι), And ( …
  -/
  simp_rw [mem_box_iff, ← _root_.le_div_iff₀' h, ← div_lt_iff₀' h]
  /-
    🎉 no goals
  -/


/-- The tag of (the index of) a `unitPartition.box`. -/
abbrev tag (ν : ι → ℤ) : ι → ℝ := fun i ↦ (ν i + 1) / n


@[simp]
theorem tag_apply (ν : ι → ℤ) (i : ι) : tag n ν i = (ν i + 1) / n := rfl


theorem tag_injective : Function.Injective (fun ν : ι → ℤ ↦ tag n ν) := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ⊢ Function.Injective fun ν => BoxIntegral.unitPartition.tag n ν
  -/
  refine fun _ _ h ↦ funext_iff.mpr fun i ↦ ?_
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    x✝¹ x✝ : ι → Int
    h : Eq ((fun ν => BoxIntegral.unitPartition.tag n ν) x✝¹) ((fun ν => BoxIntegr …
    i : ι
    ⊢ Eq (x✝¹ i) (x✝ i)
  -/
  have := congr_arg (fun x ↦ x i) h
  simp_rw [tag_apply, div_left_inj' (c := (n : ℝ)) (Nat.cast_ne_zero.mpr (NeZero.ne n)),
    add_left_inj, Int.cast_inj] at this
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    x✝¹ x✝ : ι → Int
    h : Eq ((fun ν => BoxIntegral.unitPartition.tag n ν) x✝¹) ((fun ν => BoxIntegr …
    i : ι
    this : Eq (x✝¹ i) (x✝ i)
    ⊢ Eq (x✝¹ i) (x✝ i)
  -/
  exact this
  /-
    🎉 no goals
  -/


theorem tag_mem (ν : ι → ℤ) :
    tag n ν ∈ box n ν := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    ⊢ Membership.mem (BoxIntegral.unitPartition.box n ν) (BoxIntegral.unitPartitio …
  -/
  refine mem_box_iff.mpr fun _ ↦ ?_
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x✝ : ι
    ⊢ And (LT.lt (HDiv.hDiv ↑(ν x✝) ↑n) (BoxIntegral.unitPartition.tag n ν x✝)) (L …
  -/
  rw [tag, add_div]
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x✝ : ι
    ⊢ And (LT.lt (HDiv.hDiv ↑(ν x✝) ↑n) (HAdd.hAdd (HDiv.hDiv ↑(ν x✝) ↑n) (HDiv.hD …
  -/
  have h : 0 < (n : ℝ) := Nat.cast_pos.mpr <| n.pos_of_neZero
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    x✝ : ι
    h : LT.lt 0 ↑n
    ⊢ And (LT.lt (HDiv.hDiv ↑(ν x✝) ↑n) (HAdd.hAdd (HDiv.hDiv ↑(ν x✝) ↑n) (HDiv.hD …
  -/
  exact ⟨lt_add_of_pos_right _ (by positivity), le_rfl⟩
  /-
    🎉 no goals
  -/


/-- For `x : ι → ℝ`, its index is the index of the unique `unitPartition.box` to which
it belongs. -/
def index (x : ι → ℝ) (i : ι) : ℤ := ⌈n * x i⌉ - 1


@[simp]
theorem index_apply (m : ℕ) {x : ι → ℝ} (i : ι) :
    index m x i = ⌈m * x i⌉ - 1 := rfl


variable {n} in
theorem mem_box_iff_index {x : ι → ℝ} {ν : ι → ℤ} :
    x ∈ box n ν ↔ index n x = ν := by
  simp_rw [mem_box_iff', funext_iff, index_apply, sub_eq_iff_eq_add, Int.ceil_eq_iff,
    Int.cast_add, Int.cast_one, add_sub_cancel_right]


@[simp]
theorem index_tag (ν : ι → ℤ) :
    index n (tag n ν) = ν := mem_box_iff_index.mp (tag_mem n ν)


variable {n} in
theorem disjoint {ν ν' : ι → ℤ} :
    ν ≠ ν' ↔ Disjoint (box n ν).toSet (box n ν').toSet := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν ν' : ι → Int
    ⊢ Iff (Ne ν ν') (Disjoint ↑(BoxIntegral.unitPartition.box n ν) ↑(BoxIntegral.u …
  -/
  rw [not_iff_comm, Set.not_disjoint_iff]
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν ν' : ι → Int
    ⊢ Iff (Exists fun x => And (Membership.mem (↑(BoxIntegral.unitPartition.box n  …
  -/
  refine ⟨fun ⟨x, hx, hx'⟩ ↦ ?_, fun h ↦ ⟨tag n ν, tag_mem n ν, h ▸ tag_mem n ν⟩⟩
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν ν' : ι → Int
    x✝ : Exists fun x => And (Membership.mem (↑(BoxIntegral.unitPartition.box n ν) …
    x : ι → Real
    hx : Membership.mem (↑(BoxIntegral.unitPartition.box n ν)) x
    hx' : Membership.mem (↑(BoxIntegral.unitPartition.box n ν')) x
    ⊢ Eq ν ν'
  -/
  rw [← mem_box_iff_index.mp hx, ← mem_box_iff_index.mp hx']
  /-
    🎉 no goals
  -/


theorem box_injective : Function.Injective (fun ν : ι → ℤ ↦ box n ν) := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ⊢ Function.Injective fun ν => BoxIntegral.unitPartition.box n ν
  -/
  intro _ _ h
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    a₁✝ a₂✝ : ι → Int
    h : Eq ((fun ν => BoxIntegral.unitPartition.box n ν) a₁✝) ((fun ν => BoxIntegr …
    ⊢ Eq a₁✝ a₂✝
  -/
  contrapose! h
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    a₁✝ a₂✝ : ι → Int
    h : Ne a₁✝ a₂✝
    ⊢ Ne (BoxIntegral.unitPartition.box n a₁✝) (BoxIntegral.unitPartition.box n a₂✝)
  -/
  exact Box.ne_of_disjoint_coe (disjoint.mp h)
  /-
    🎉 no goals
  -/


lemma box.upper_sub_lower (ν : ι → ℤ) (i : ι) :
    (box n ν ).upper i - (box n ν).lower i = 1 / n := by
  /-
    ι : Type u_1
    n : Nat
    inst✝ : NeZero n
    ν : ι → Int
    i : ι
    ⊢ Eq (HSub.hSub ((BoxIntegral.unitPartition.box n ν).upper i) ((BoxIntegral.un …
  -/
  simp_rw [box, add_div, add_sub_cancel_left]
  /-
    🎉 no goals
  -/


theorem diam_boxIcc (ν : ι → ℤ) :
    Metric.diam (Box.Icc (box n ν)) ≤ 1 / n := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    ⊢ LE.le (Metric.diam (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) …
  -/
  rw [BoxIntegral.Box.Icc_eq_pi]
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    ⊢ LE.le (Metric.diam (Set.univ.pi fun i => Set.Icc ((BoxIntegral.unitPartition …
  -/
  refine ENNReal.toReal_le_of_le_ofReal (by positivity) <| EMetric.diam_pi_le_of_le (fun i ↦ ?_)
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    i : ι
    ⊢ LE.le (EMetric.diam (Set.Icc ((BoxIntegral.unitPartition.box n ν).lower i) ( …
  -/
  simp_rw [Real.ediam_Icc, box.upper_sub_lower, le_rfl]
  /-
    🎉 no goals
  -/


@[simp]
theorem volume_box (ν : ι → ℤ) :
    volume (box n ν : Set (ι → ℝ)) = 1 / n ^ card ι := by
  simp_rw [volume_pi, BoxIntegral.Box.coe_eq_pi, Measure.pi_pi, Real.volume_Ioc,
    box.upper_sub_lower, Finset.prod_const, ENNReal.ofReal_div_of_pos (Nat.cast_pos.mpr
    n.pos_of_neZero), ENNReal.ofReal_one, ENNReal.ofReal_natCast, one_div, ENNReal.inv_pow,
    Finset.card_univ]


                                                 /-
                                                   ι : Type u_1
                                                   n : Nat
                                                   inst✝¹ : NeZero n
                                                   inst✝ : Fintype ι
                                                   s : Set (ι → Real)
                                                   ⊢ MeasureTheory.Measure (ι → Real)
                                                 -/
theorem setFinite_index {s : Set (ι → ℝ)} (hs₁ : NullMeasurableSet s) (hs₂ : volume s ≠ ⊤) :
                                                 /-
                                                   🎉 no goals
                                                 -/
    Set.Finite {ν : ι → ℤ | ↑(box n ν) ⊆ s} := by
  refine (Measure.finite_const_le_meas_of_disjoint_iUnion₀ volume (ε := 1 / n ^ card ι)
    (by norm_num) (As := fun ν : ι → ℤ ↦ (box n ν) ∩ s) (fun ν ↦ ?_) (fun _ _ h ↦ ?_) ?_).subset
      (fun _ hν ↦ ?_)
    /-
      case refine_1
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
      hs₂ : Ne (MeasureTheory.MeasureSpace.volume s) Top.top
      ν : ι → Int
      ⊢ MeasureTheory.NullMeasurableSet ((fun ν => Inter.inter (↑(BoxIntegral.unitPa …
    -/
  · refine NullMeasurableSet.inter ?_ hs₁
    /-
      case refine_1
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
      hs₂ : Ne (MeasureTheory.MeasureSpace.volume s) Top.top
      ν : ι → Int
      ⊢ MeasureTheory.NullMeasurableSet (↑(BoxIntegral.unitPartition.box n ν)) Measu …
    -/
    exact (box n ν).measurableSet_coe.nullMeasurableSet
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
      hs₂ : Ne (MeasureTheory.MeasureSpace.volume s) Top.top
      x✝¹ x✝ : ι → Int
      h : Ne x✝¹ x✝
      ⊢ Function.onFun (MeasureTheory.AEDisjoint MeasureTheory.MeasureSpace.volume)  …
    -/
  · exact ((Disjoint.inter_right _ (disjoint.mp h)).inter_left _ ).aedisjoint
    /-
      🎉 no goals
    -/
  · exact lt_top_iff_ne_top.mp <| measure_lt_top_of_subset
      (by simp only [Set.iUnion_subset_iff, Set.inter_subset_right, implies_true]) hs₂
    /-
      case refine_4
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
      hs₂ : Ne (MeasureTheory.MeasureSpace.volume s) Top.top
      x✝ : ι → Int
      hν : Membership.mem (setOf fun ν => HasSubset.Subset (↑(BoxIntegral.unitPartit …
      ⊢ Membership.mem (setOf fun i => LE.le (HDiv.hDiv 1 (HPow.hPow (↑n) (Fintype.c …
    -/
  · rw [Set.mem_setOf, Set.inter_eq_self_of_subset_left hν, volume_box]
    /-
      🎉 no goals
    -/


/-- For `B : BoxIntegral.Box`, the set of indices of `unitPartition.box` that are subsets of `B`.
This is a finite set. These boxes cover `B` if it has integral vertices, see
`unitPartition.prepartition_isPartition`. -/
def admissibleIndex (B : Box ι) : Finset (ι → ℤ) := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ⊢ Finset (ι → Int)
  -/
  refine (setFinite_index n B.measurableSet_coe.nullMeasurableSet ?_).toFinset
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ⊢ Ne (MeasureTheory.MeasureSpace.volume ↑B) Top.top
  -/
  exact lt_top_iff_ne_top.mp (IsBounded.measure_lt_top B.isBounded)
  /-
    🎉 no goals
  -/


variable {n} in
theorem mem_admissibleIndex_iff {B : Box ι} {ν : ι → ℤ} :
    ν ∈ admissibleIndex n B ↔ box n ν ≤ B := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ν : ι → Int
    ⊢ Iff (Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν) (LE.l …
  -/
  rw [admissibleIndex, Set.Finite.mem_toFinset, Set.mem_setOf_eq, Box.coe_subset_coe]
  /-
    🎉 no goals
  -/


open Classical in
/-- For `B : BoxIntegral.Box`, the `TaggedPrepartition` formed by the set of all
`unitPartition.box` whose index is `B`-admissible. -/
def prepartition (B : Box ι) : TaggedPrepartition B where
  boxes := Finset.image (fun ν ↦ box n ν) (admissibleIndex n B)
  le_of_mem' _ hI := by
    /-
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B x✝ : BoxIntegral.Box ι
      hI : Membership.mem (Finset.image (fun ν => BoxIntegral.unitPartition.box n ν) …
      ⊢ LE.le x✝ B
    -/
    obtain ⟨_, hν, rfl⟩ := Finset.mem_image.mp hI
    /-
      case intro.intro
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      w✝ : ι → Int
      hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) w✝
      hI : Membership.mem (Finset.image (fun ν => BoxIntegral.unitPartition.box n ν) …
      ⊢ LE.le (BoxIntegral.unitPartition.box n w✝) B
    -/
    exact mem_admissibleIndex_iff.mp hν
    /-
      🎉 no goals
    -/
  pairwiseDisjoint _ hI₁ _ hI₂ h := by
    /-
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B x✝¹ : BoxIntegral.Box ι
      hI₁ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      x✝ : BoxIntegral.Box ι
      hI₂ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      h : Ne x✝¹ x✝
      ⊢ Function.onFun Disjoint BoxIntegral.Box.toSet x✝¹ x✝
    -/
    obtain ⟨_, _, rfl⟩ := Finset.mem_image.mp hI₁
    /-
      case intro.intro
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B x✝ : BoxIntegral.Box ι
      hI₂ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      w✝ : ι → Int
      left✝ : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) w✝
      hI₁ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      h : Ne (BoxIntegral.unitPartition.box n w✝) x✝
      ⊢ Function.onFun Disjoint BoxIntegral.Box.toSet (BoxIntegral.unitPartition.box …
    -/
    obtain ⟨_, _, rfl⟩ := Finset.mem_image.mp hI₂
    /-
      case intro.intro.intro.intro
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      w✝¹ : ι → Int
      left✝¹ : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) w✝¹
      hI₁ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      w✝ : ι → Int
      left✝ : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) w✝
      hI₂ : Membership.mem (↑(Finset.image (fun ν => BoxIntegral.unitPartition.box n …
      h : Ne (BoxIntegral.unitPartition.box n w✝¹) (BoxIntegral.unitPartition.box n  …
      ⊢ Function.onFun Disjoint BoxIntegral.Box.toSet (BoxIntegral.unitPartition.box …
    -/
    exact disjoint.mp fun x ↦ h (congrArg (box n) x)
    /-
      🎉 no goals
    -/
  tag I :=
    if hI : ∃ ν ∈ admissibleIndex n B, I = box n ν then tag n hI.choose else B.exists_mem.choose
  tag_mem_Icc I := by
    /-
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B I : BoxIntegral.Box ι
      ⊢ Membership.mem (BoxIntegral.Box.Icc B) ((fun I => dite (Exists fun ν => And  …
    -/
    by_cases hI : ∃ ν ∈ admissibleIndex n B, I = box n ν
      /-
        case pos
        ι : Type u_1
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : Fintype ι
        B I : BoxIntegral.Box ι
        hI : Exists fun ν => And (Membership.mem (BoxIntegral.unitPartition.admissible …
        ⊢ Membership.mem (BoxIntegral.Box.Icc B) ((fun I => dite (Exists fun ν => And  …
      -/
    · simp_rw [dif_pos hI]
      /-
        case pos
        ι : Type u_1
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : Fintype ι
        B I : BoxIntegral.Box ι
        hI : Exists fun ν => And (Membership.mem (BoxIntegral.unitPartition.admissible …
        ⊢ Membership.mem (BoxIntegral.Box.Icc B) (BoxIntegral.unitPartition.tag n hI.c …
      -/
      exact Box.coe_subset_Icc <| (mem_admissibleIndex_iff.mp hI.choose_spec.1) (tag_mem n _)
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : Fintype ι
        B I : BoxIntegral.Box ι
        hI : Not (Exists fun ν => And (Membership.mem (BoxIntegral.unitPartition.admis …
        ⊢ Membership.mem (BoxIntegral.Box.Icc B) ((fun I => dite (Exists fun ν => And  …
      -/
    · simp_rw [dif_neg hI]
      /-
        case neg
        ι : Type u_1
        n : Nat
        inst✝¹ : NeZero n
        inst✝ : Fintype ι
        B I : BoxIntegral.Box ι
        hI : Not (Exists fun ν => And (Membership.mem (BoxIntegral.unitPartition.admis …
        ⊢ Membership.mem (BoxIntegral.Box.Icc B) ⋯.choose
      -/
      exact Box.coe_subset_Icc B.exists_mem.choose_spec
      /-
        🎉 no goals
      -/


variable {n} in
@[simp]
theorem mem_prepartition_iff {B I : Box ι} :
    I ∈ prepartition n B ↔ ∃ ν ∈ admissibleIndex n B, box n ν = I := by
  classical
  rw [prepartition, TaggedPrepartition.mem_mk, Prepartition.mem_mk, Finset.mem_image]


variable {n} in
theorem mem_prepartition_boxes_iff {B I : Box ι} :
    I ∈ (prepartition n B).boxes ↔ ∃ ν ∈ admissibleIndex n B, box n ν = I :=
  mem_prepartition_iff


theorem prepartition_tag {ν : ι → ℤ} {B : Box ι} (hν : ν ∈ admissibleIndex n B) :
    (prepartition n B).tag (box n ν) = tag n ν := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    B : BoxIntegral.Box ι
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    ⊢ Eq ((BoxIntegral.unitPartition.prepartition n B).tag (BoxIntegral.unitPartit …
  -/
  dsimp only [prepartition]
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    B : BoxIntegral.Box ι
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    ⊢ Eq (dite (Exists fun ν_1 => And (Membership.mem (BoxIntegral.unitPartition.a …
  -/
  have h : ∃ ν' ∈ admissibleIndex n B, box n ν = box n ν' := ⟨ν, hν, rfl⟩
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    B : BoxIntegral.Box ι
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    h : Exists fun ν' => And (Membership.mem (BoxIntegral.unitPartition.admissible …
    ⊢ Eq (dite (Exists fun ν_1 => And (Membership.mem (BoxIntegral.unitPartition.a …
  -/
  rw [dif_pos h, (tag_injective n).eq_iff, ← (box_injective n).eq_iff]
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    B : BoxIntegral.Box ι
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    h : Exists fun ν' => And (Membership.mem (BoxIntegral.unitPartition.admissible …
    ⊢ Eq (BoxIntegral.unitPartition.box n h.choose) (BoxIntegral.unitPartition.box …
  -/
  exact h.choose_spec.2.symm
  /-
    🎉 no goals
  -/


theorem box_index_tag_eq_self {B I : Box ι} (hI : I ∈ (prepartition n B).boxes) :
    box n (index n ((prepartition n B).tag I)) = I := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B I : BoxIntegral.Box ι
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B).boxes I
    ⊢ Eq (BoxIntegral.unitPartition.box n (BoxIntegral.unitPartition.index n ((Box …
  -/
  obtain ⟨ν, hν, rfl⟩ := mem_prepartition_boxes_iff.mp hI
  /-
    case intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ν : ι → Int
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B).boxes (BoxInt …
    ⊢ Eq (BoxIntegral.unitPartition.box n (BoxIntegral.unitPartition.index n ((Box …
  -/
  rw [prepartition_tag n hν, index_tag]
  /-
    🎉 no goals
  -/


theorem prepartition_isHenstock (B : Box ι) :
    (prepartition n B).IsHenstock := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ⊢ (BoxIntegral.unitPartition.prepartition n B).IsHenstock
  -/
  intro _ hI
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B J✝ : BoxIntegral.Box ι
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) J✝
    ⊢ Membership.mem (BoxIntegral.Box.Icc J✝) ((BoxIntegral.unitPartition.preparti …
  -/
  obtain ⟨ν, hν, rfl⟩ := mem_prepartition_iff.mp hI
  /-
    case intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ν : ι → Int
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
    ⊢ Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) ((B …
  -/
  rw [prepartition_tag n hν]
  /-
    case intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    ν : ι → Int
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
    ⊢ Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) (Bo …
  -/
  exact Box.coe_subset_Icc (tag_mem _ _)
  /-
    🎉 no goals
  -/


theorem prepartition_isSubordinate (B : Box ι) {r : ℝ} (hr : 0 < r) (hn : 1 / n ≤ r) :
    (prepartition n B).IsSubordinate (fun _ ↦ ⟨r, hr⟩) := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    r : Real
    hr : LT.lt 0 r
    hn : LE.le (HDiv.hDiv 1 ↑n) r
    ⊢ (BoxIntegral.unitPartition.prepartition n B).IsSubordinate fun x => ⟨r, hr⟩
  -/
  intro _ hI
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    r : Real
    hr : LT.lt 0 r
    hn : LE.le (HDiv.hDiv 1 ↑n) r
    J✝ : BoxIntegral.Box ι
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) J✝
    ⊢ HasSubset.Subset (BoxIntegral.Box.Icc J✝) (Metric.closedBall ((BoxIntegral.u …
  -/
  obtain ⟨ν, hν, rfl⟩ := mem_prepartition_iff.mp hI
  /-
    case intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    r : Real
    hr : LT.lt 0 r
    hn : LE.le (HDiv.hDiv 1 ↑n) r
    ν : ι → Int
    hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
    hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
    ⊢ HasSubset.Subset (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) ( …
  -/
  refine fun _ h ↦ le_trans (Metric.dist_le_diam_of_mem (Box.isBounded_Icc _) h ?_) ?_
    /-
      case intro.intro.refine_1
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      r : Real
      hr : LT.lt 0 r
      hn : LE.le (HDiv.hDiv 1 ↑n) r
      ν : ι → Int
      hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
      hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
      x✝ : ι → Real
      h : Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) x✝
      ⊢ Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) ((B …
    -/
  · rw [prepartition_tag n hν]
    /-
      case intro.intro.refine_1
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      r : Real
      hr : LT.lt 0 r
      hn : LE.le (HDiv.hDiv 1 ↑n) r
      ν : ι → Int
      hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
      hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
      x✝ : ι → Real
      h : Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) x✝
      ⊢ Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) (Bo …
    -/
    exact Box.coe_subset_Icc (tag_mem _ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      r : Real
      hr : LT.lt 0 r
      hn : LE.le (HDiv.hDiv 1 ↑n) r
      ν : ι → Int
      hν : Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) ν
      hI : Membership.mem (BoxIntegral.unitPartition.prepartition n B) (BoxIntegral. …
      x✝ : ι → Real
      h : Membership.mem (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) x✝
      ⊢ LE.le (Metric.diam (BoxIntegral.Box.Icc (BoxIntegral.unitPartition.box n ν)) …
    -/
  · exact le_trans (diam_boxIcc n ν) hn
    /-
      🎉 no goals
    -/


private theorem mem_admissibleIndex_of_mem_box_aux₁ (x : ℝ) (a : ℤ) :
    a < x ↔ a ≤ (⌈n * x⌉ - 1) / (n : ℝ) := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : Real
    a : Int
    ⊢ Iff (LT.lt (↑a) x) (LE.le (↑a) (HDiv.hDiv (HSub.hSub (↑(Int.ceil (HMul.hMul  …
  -/
  have h : 0 < (n : ℝ) := Nat.cast_pos.mpr <| n.pos_of_neZero
  rw [le_div_iff₀' h, le_sub_iff_add_le,
    show (n : ℝ) * a + 1 = (n * a + 1 : ℤ) by norm_cast,
    Int.cast_le, Int.add_one_le_ceil_iff, Int.cast_mul, Int.cast_natCast, mul_lt_mul_left h]


private theorem mem_admissibleIndex_of_mem_box_aux₂ (x : ℝ) (a : ℤ) :
    x ≤ a ↔ (⌈n * x⌉ - 1 + 1) / (n : ℝ) ≤ a := by
  /-
    n : Nat
    inst✝ : NeZero n
    x : Real
    a : Int
    ⊢ Iff (LE.le x ↑a) (LE.le (HDiv.hDiv (HAdd.hAdd (HSub.hSub (↑(Int.ceil (HMul.h …
  -/
  have h : 0 < (n : ℝ) := Nat.cast_pos.mpr <| n.pos_of_neZero
  rw [sub_add_cancel, div_le_iff₀' h,
    show (n : ℝ) * a = (n * a : ℤ) by norm_cast,
    Int.cast_le, Int.ceil_le, Int.cast_mul, Int.cast_natCast, mul_le_mul_left h]


/-- If `B : BoxIntegral.Box` has integral vertices and contains the point `x`, then the index of
`x` is admissible for `B`. -/
theorem mem_admissibleIndex_of_mem_box {B : Box ι} (hB : hasIntegralVertices B) {x : ι → ℝ}
    (hx : x ∈ B) : index n x ∈ admissibleIndex n B := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    x : ι → Real
    hx : Membership.mem B x
    ⊢ Membership.mem (BoxIntegral.unitPartition.admissibleIndex n B) (BoxIntegral. …
  -/
  obtain ⟨l, u, hl, hu⟩ := hB
  simp_rw [mem_admissibleIndex_iff, Box.le_iff_bounds, box_lower, box_upper, Pi.le_def,
    index_apply, hl, hu, ← forall_and]
  /-
    case intro.intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    x : ι → Real
    hx : Membership.mem B x
    l u : ι → Int
    hl : ∀ (i : ι), Eq (B.lower i) ↑(l i)
    hu : ∀ (i : ι), Eq (B.upper i) ↑(u i)
    ⊢ ∀ (x_1 : ι), And (LE.le (↑(l x_1)) (HDiv.hDiv ↑(HSub.hSub (Int.ceil (HMul.hM …
  -/
  push_cast
  /-
    case intro.intro.intro
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    x : ι → Real
    hx : Membership.mem B x
    l u : ι → Int
    hl : ∀ (i : ι), Eq (B.lower i) ↑(l i)
    hu : ∀ (i : ι), Eq (B.upper i) ↑(u i)
    ⊢ ∀ (x_1 : ι), And (LE.le (↑(l x_1)) (HDiv.hDiv (HSub.hSub (↑(Int.ceil (HMul.h …
  -/
  refine fun i ↦ ⟨?_, ?_⟩
    /-
      case intro.intro.intro.refine_1
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      x : ι → Real
      hx : Membership.mem B x
      l u : ι → Int
      hl : ∀ (i : ι), Eq (B.lower i) ↑(l i)
      hu : ∀ (i : ι), Eq (B.upper i) ↑(u i)
      i : ι
      ⊢ LE.le (↑(l i)) (HDiv.hDiv (HSub.hSub (↑(Int.ceil (HMul.hMul (↑n) (x i)))) 1) …
    -/
  · exact (mem_admissibleIndex_of_mem_box_aux₁ n (x i) (l i)).mp ((hl i) ▸ (hx i).1)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.refine_2
      ι : Type u_1
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Fintype ι
      B : BoxIntegral.Box ι
      x : ι → Real
      hx : Membership.mem B x
      l u : ι → Int
      hl : ∀ (i : ι), Eq (B.lower i) ↑(l i)
      hu : ∀ (i : ι), Eq (B.upper i) ↑(u i)
      i : ι
      ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HSub.hSub (↑(Int.ceil (HMul.hMul (↑n) (x i))))  …
    -/
  · exact (mem_admissibleIndex_of_mem_box_aux₂ n (x i) (u i)).mp ((hu i) ▸ (hx i).2)
    /-
      🎉 no goals
    -/


/-- If `B : BoxIntegral.Box` has integral vertices, then `prepartition n B` is a partition of
`B`. -/
theorem prepartition_isPartition {B : Box ι} (hB : hasIntegralVertices B) :
    (prepartition n B).IsPartition := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    ⊢ (BoxIntegral.unitPartition.prepartition n B).IsPartition
  -/
  refine fun x hx ↦ ⟨box n (index n x), ?_, mem_box_iff_index.mpr rfl⟩
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    x : ι → Real
    hx : Membership.mem B x
    ⊢ Membership.mem (BoxIntegral.unitPartition.prepartition n B).toPrepartition ( …
  -/
  rw [TaggedPrepartition.mem_toPrepartition, mem_prepartition_iff]
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    x : ι → Real
    hx : Membership.mem B x
    ⊢ Exists fun ν => And (Membership.mem (BoxIntegral.unitPartition.admissibleInd …
  -/
  exact ⟨index n x, mem_admissibleIndex_of_mem_box n hB hx, rfl⟩
  /-
    🎉 no goals
  -/


local notation "L" => span ℤ (Set.range (Pi.basisFun ℝ ι))


variable {n} in
theorem mem_smul_span_iff {v : ι → ℝ} :
    v ∈ (n : ℝ)⁻¹ • L ↔ ∀ i, n * v i ∈ Set.range (algebraMap ℤ ℝ) := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    v : ι → Real
    ⊢ Iff (Membership.mem (HSMul.hSMul (Inv.inv ↑n) (Submodule.span Int (Set.range …
  -/
  rw [ZSpan.smul _ (inv_ne_zero (NeZero.ne _)), Basis.mem_span_iff_repr_mem]
  simp_rw [Basis.repr_isUnitSMul, Pi.basisFun_repr, Units.smul_def, Units.val_inv_eq_inv_val,
    IsUnit.unit_spec, inv_inv, smul_eq_mul]


theorem tag_mem_smul_span (ν : ι → ℤ) :
    tag n ν ∈ (n : ℝ)⁻¹ • L := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    ν : ι → Int
    ⊢ Membership.mem (HSMul.hSMul (Inv.inv ↑n) (Submodule.span Int (Set.range ⇑(Pi …
  -/
  refine mem_smul_span_iff.mpr fun i ↦ ⟨ν i + 1, ?_⟩
  rw [tag_apply, div_eq_inv_mul, ← mul_assoc, mul_inv_cancel_of_invertible, one_mul, map_add,
    map_one, eq_intCast]


theorem tag_index_eq_self_of_mem_smul_span {x : ι → ℝ} (hx : x ∈ (n : ℝ)⁻¹ • L) :
    tag n (index n x) = x := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    x : ι → Real
    hx : Membership.mem (HSMul.hSMul (Inv.inv ↑n) (Submodule.span Int (Set.range ⇑ …
    ⊢ Eq (BoxIntegral.unitPartition.tag n (BoxIntegral.unitPartition.index n x)) x
  -/
  rw [mem_smul_span_iff] at hx
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    x : ι → Real
    hx : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap Int Real)) (HMul.hMul ( …
    ⊢ Eq (BoxIntegral.unitPartition.tag n (BoxIntegral.unitPartition.index n x)) x
  -/
  ext i
  /-
    case h
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    x : ι → Real
    hx : ∀ (i : ι), Membership.mem (Set.range ⇑(algebraMap Int Real)) (HMul.hMul ( …
    i : ι
    ⊢ Eq (BoxIntegral.unitPartition.tag n (BoxIntegral.unitPartition.index n x) i) …
  -/
  obtain ⟨a, ha⟩ : ∃ a : ℤ, a = n * x i := hx i
  rwa [tag_apply, index_apply, Int.cast_sub, Int.cast_one, sub_add_cancel, ← ha, Int.ceil_intCast,
    div_eq_iff (NeZero.ne _), mul_comm]


theorem eq_of_mem_smul_span_of_index_eq_index {x y : ι → ℝ} (hx : x ∈ (n : ℝ)⁻¹ • L)
    (hy : y ∈ (n : ℝ)⁻¹ • L) (h : index n x = index n y) : x = y := by
  /-
    ι : Type u_1
    n : Nat
    inst✝¹ : NeZero n
    inst✝ : Fintype ι
    x y : ι → Real
    hx : Membership.mem (HSMul.hSMul (Inv.inv ↑n) (Submodule.span Int (Set.range ⇑ …
    hy : Membership.mem (HSMul.hSMul (Inv.inv ↑n) (Submodule.span Int (Set.range ⇑ …
    h : Eq (BoxIntegral.unitPartition.index n x) (BoxIntegral.unitPartition.index  …
    ⊢ Eq x y
  -/
  rw [← tag_index_eq_self_of_mem_smul_span n hx, ← tag_index_eq_self_of_mem_smul_span n hy, h]
  /-
    🎉 no goals
  -/


theorem integralSum_eq_tsum_div {B : Box ι} (hB : hasIntegralVertices B) (hs₀ : s ≤ B) :
    integralSum (Set.indicator s F) (BoxAdditiveMap.toSMul (Measure.toBoxAdditive volume))
      (prepartition n B) = (∑' x : ↑(s ∩ (n : ℝ)⁻¹ • L), F x) / n ^ card ι := by
  classical
  unfold integralSum
  have : Fintype ↑(s ∩ (n : ℝ)⁻¹ • L) := by
    apply Set.Finite.fintype
    rw [← coe_pointwise_smul, ZSpan.smul _ (inv_ne_zero (NeZero.ne _))]
    exact ZSpan.setFinite_inter _ (B.isBounded.subset hs₀)
  rw [tsum_fintype, Finset.sum_set_coe, Finset.sum_div, eq_comm]
  simp_rw [Set.indicator_apply, apply_ite, BoxAdditiveMap.toSMul_apply, Measure.toBoxAdditive_apply,
    smul_eq_mul, mul_zero, Finset.sum_ite, Finset.sum_const_zero, add_zero]
  refine Finset.sum_bij (fun x _ ↦ box n (index n x)) (fun _ hx ↦ Finset.mem_filter.mpr ?_)
    (fun _ hx _ hy h ↦ ?_) (fun I hI ↦ ?_) (fun _ hx ↦ ?_)
  · rw [Set.mem_toFinset] at hx
    refine ⟨mem_prepartition_boxes_iff.mpr
      ⟨index n _, mem_admissibleIndex_of_mem_box n hB (hs₀ hx.1), rfl⟩, ?_⟩
    simp_rw [prepartition_tag n (mem_admissibleIndex_of_mem_box n hB (hs₀ hx.1)),
      tag_index_eq_self_of_mem_smul_span n hx.2, hx.1]
  · rw [Set.mem_toFinset] at hx hy
    exact eq_of_mem_smul_span_of_index_eq_index n hx.2 hy.2 (box_injective n h)
  · rw [Finset.mem_filter] at hI
    refine ⟨(prepartition n B).tag I, Set.mem_toFinset.mpr ⟨hI.2, ?_⟩, box_index_tag_eq_self n hI.1⟩
    rw [← box_index_tag_eq_self n hI.1, prepartition_tag n
      (mem_admissibleIndex_of_mem_box n hB (hs₀ hI.2))]
    exact tag_mem_smul_span _ _
  · rw [Set.mem_toFinset] at hx
    rw [volume_box, prepartition_tag n (mem_admissibleIndex_of_mem_box n hB (hs₀ hx.1)),
      tag_index_eq_self_of_mem_smul_span n hx.2, ENNReal.toReal_div,
      ENNReal.one_toReal, ENNReal.toReal_pow, ENNReal.toReal_nat, mul_comm_div, one_mul]


/-- Let `s` be a bounded, measurable set of `ι → ℝ` whose frontier has zero volume and let `F`
be a continuous function. Then the limit as `n → ∞` of `∑ F x / n ^ card ι`, where the sum is
over the points in `s ∩ n⁻¹ • (ι → ℤ)`, tends to the integral of `F` over `s`. -/
theorem _root_.tendsto_tsum_div_pow_atTop_integral (hF : Continuous F) (hs₁ : IsBounded s)
    (hs₂ : MeasurableSet s) (hs₃ : volume (frontier s) = 0) :
    Tendsto (fun n : ℕ ↦ (∑' x : ↑(s ∩ (n : ℝ)⁻¹ • L), F x) / n ^ card ι)
      atTop (nhds (∫ x in s, F x)) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    F : (ι → Real) → Real
    hF : Continuous F
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (tsum fun x => F ↑x) (HPow.hPow (↑n) (Fin …
  -/
  obtain ⟨B, hB, hs₀⟩ := le_hasIntegralVertices_of_isBounded hs₁
  /-
    case intro.intro
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    F : (ι → Real) → Real
    hF : Continuous F
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    hs₀ : LE.le s ↑B
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (tsum fun x => F ↑x) (HPow.hPow (↑n) (Fin …
  -/
  refine Metric.tendsto_atTop.mpr fun ε hε ↦ ?_
  have h₁ : ∃ C, ∀ x ∈ Box.Icc B, ‖Set.indicator s F x‖ ≤ C := by
    obtain ⟨C₀, h₀⟩ := (Box.isCompact_Icc B).exists_bound_of_continuousOn hF.continuousOn
    refine ⟨max 0 C₀, fun x hx ↦ ?_⟩
    rw [Set.indicator]
    split_ifs with hs
    · exact le_max_of_le_right (h₀ x hx)
    · exact norm_zero.trans_le <|le_max_left 0 _
  have h₂ : ∀ᵐ x, ContinuousAt (s.indicator F) x := by
    filter_upwards [compl_mem_ae_iff.mpr hs₃] with _ h
      using (hF.continuousOn).continuousAt_indicator h
  obtain ⟨r, hr₁, hr₂⟩ := (hasIntegral_iff.mp <|
      AEContinuous.hasBoxIntegral (volume : Measure (ι → ℝ)) h₁ h₂
        IntegrationParams.Riemann) (ε / 2) (half_pos hε)
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    F : (ι → Real) → Real
    hF : Continuous F
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    hs₀ : LE.le s ↑B
    ε : Real
    hε : GT.gt ε 0
    h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
    h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
    hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
    hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
    ⊢ Exists fun N => ∀ (n : Nat), GE.ge n N → LT.lt (Dist.dist (HDiv.hDiv (tsum f …
  -/
  refine ⟨⌈(r 0 0 : ℝ)⁻¹⌉₊, fun n hn ↦ lt_of_le_of_lt ?_ (half_lt_self_iff.mpr hε)⟩
  have : NeZero n :=
    ⟨Nat.ne_zero_iff_zero_lt.mpr <| (Nat.ceil_pos.mpr (inv_pos.mpr (r 0 0).prop)).trans_le hn⟩
  rw [← integralSum_eq_tsum_div _ s F hB hs₀, ← Measure.restrict_restrict_of_subset hs₀,
    ← integral_indicator hs₂]
  /-
    case intro.intro.intro.intro
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    F : (ι → Real) → Real
    hF : Continuous F
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    B : BoxIntegral.Box ι
    hB : BoxIntegral.hasIntegralVertices B
    hs₀ : LE.le s ↑B
    ε : Real
    hε : GT.gt ε 0
    h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
    h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
    r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
    hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
    hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
    n : Nat
    hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
    this : NeZero n
    ⊢ LE.le (Dist.dist (BoxIntegral.integralSum (s.indicator F) MeasureTheory.Meas …
  -/
  refine hr₂ 0 _ ⟨?_, fun _ ↦ ?_, fun h ↦ ?_, fun h ↦ ?_⟩ (prepartition_isPartition _ hB)
    /-
      case intro.intro.intro.intro.refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      ⊢ (BoxIntegral.unitPartition.prepartition n B).IsSubordinate (r 0)
    -/
  · rw [show r 0 = fun _ ↦ r 0 0 from funext_iff.mpr (hr₁ 0 rfl)]
    /-
      case intro.intro.intro.intro.refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      ⊢ (BoxIntegral.unitPartition.prepartition n B).IsSubordinate fun x => r 0 0
    -/
    apply prepartition_isSubordinate n B
    /-
      case intro.intro.intro.intro.refine_1.hn
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      ⊢ LE.le (HDiv.hDiv 1 ↑n) ↑(r 0 0)
    -/
    rw [one_div, inv_le_comm₀ (mod_cast (Nat.pos_of_neZero n)) (r 0 0).prop]
    /-
      case intro.intro.intro.intro.refine_1.hn
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      ⊢ LE.le (Inv.inv ↑(r 0 0)) ↑n
    -/
    exact le_trans (Nat.le_ceil _) (Nat.cast_le.mpr hn)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      x✝ : Eq BoxIntegral.IntegrationParams.Riemann.bHenstock Bool.true
      ⊢ (BoxIntegral.unitPartition.prepartition n B).IsHenstock
    -/
  · exact prepartition_isHenstock n B
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_3
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      h : Eq BoxIntegral.IntegrationParams.Riemann.bDistortion Bool.true
      ⊢ LE.le (BoxIntegral.unitPartition.prepartition n B).distortion 0
    -/
  · simp only [IntegrationParams.Riemann, Bool.false_eq_true] at h
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.refine_4
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      F : (ι → Real) → Real
      hF : Continuous F
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      B : BoxIntegral.Box ι
      hB : BoxIntegral.hasIntegralVertices B
      hs₀ : LE.le s ↑B
      ε : Real
      hε : GT.gt ε 0
      h₁ : Exists fun C => ∀ (x : ι → Real), Membership.mem (BoxIntegral.Box.Icc B)  …
      h₂ : Filter.Eventually (fun x => ContinuousAt (s.indicator F) x) (MeasureTheor …
      r : NNReal → (ι → Real) → ↑(Set.Ioi 0)
      hr₁ : ∀ (c : NNReal), BoxIntegral.IntegrationParams.Riemann.RCond (r c)
      hr₂ : ∀ (c : NNReal) (π : BoxIntegral.TaggedPrepartition B), BoxIntegral.Integ …
      n : Nat
      hn : GE.ge n (Nat.ceil (Inv.inv ↑(r 0 0)))
      this : NeZero n
      h : Eq BoxIntegral.IntegrationParams.Riemann.bDistortion Bool.true
      ⊢ Exists fun π' => And (Eq π'.iUnion (SDiff.sdiff (↑B) (BoxIntegral.unitPartit …
    -/
  · simp only [IntegrationParams.Riemann, Bool.false_eq_true] at h
    /-
      🎉 no goals
    -/


/-- Let `s` be a bounded, measurable set of `ι → ℝ` whose frontier has zero volume. Then the limit
as `n → ∞` of `card (s ∩ n⁻¹ • (ι → ℤ)) / n ^ card ι` tends to the volume of `s`. This is a
special case of `tendsto_card_div_pow` with `F = 1`. -/
theorem _root_.tendsto_card_div_pow_atTop_volume (hs₁ : IsBounded s)
    (hs₂ : MeasurableSet s) (hs₃ : volume (frontier s) = 0) :
    Tendsto (fun n : ℕ ↦ (Nat.card ↑(s ∩ (n : ℝ)⁻¹ • L) : ℝ) / n ^ card ι)
      atTop (nhds (volume s).toReal) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    ⊢ Filter.Tendsto (fun n => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
  -/
  convert tendsto_tsum_div_pow_atTop_integral s (fun _ ↦ 1) continuous_const hs₁ hs₂ hs₃
    /-
      case h.e'_3.h.h.e'_5
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      x✝ : Nat
      ⊢ Eq (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑x✝) ↑(Submodule.span I …
    -/
  · rw [tsum_const, nsmul_eq_mul, mul_one, Nat.cast_inj]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_5.h.e'_3
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      ⊢ Eq (MeasureTheory.MeasureSpace.volume s).toReal (MeasureTheory.integral (Mea …
    -/
  · rw [setIntegral_const, smul_eq_mul, mul_one]
    /-
      🎉 no goals
    -/


private def tendsto_card_div_pow₁ {c : ℝ} (hc : c ≠ 0) :
    ↑(s ∩ c⁻¹ • L) ≃ ↑(c • s ∩ L) :=
  Equiv.subtypeEquiv (Equiv.smulRight hc) (fun x ↦ by
    simp_rw [Set.mem_inter_iff, Equiv.smulRight_apply, Set.smul_mem_smul_set_iff₀ hc,
      ← Set.mem_inv_smul_set_iff₀ hc])


private theorem tendsto_card_div_pow₂ (hs₁ : IsBounded s)
    (hs₄ : ∀ ⦃x y : ℝ⦄, 0 < x → x ≤ y → x • s ⊆ y • s) {x y : ℝ} (hx : 0 < x) (hy : x ≤ y) :
    Nat.card ↑(s ∩ x⁻¹ • L) ≤ Nat.card ↑(s ∩ y⁻¹ • L) := by
  rw [Nat.card_congr (tendsto_card_div_pow₁ s hx.ne'),
      Nat.card_congr (tendsto_card_div_pow₁ s (hx.trans_le hy).ne')]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    x y : Real
    hx : LT.lt 0 x
    hy : LE.le x y
    ⊢ LE.le (Nat.card ↑(Inter.inter (HSMul.hSMul x s) ↑(Submodule.span Int (Set.ra …
  -/
  refine Nat.card_mono ?_ ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le x y
      ⊢ (Inter.inter (HSMul.hSMul y s) ↑(Submodule.span Int (Set.range ⇑(Pi.basisFun …
    -/
  · exact ZSpan.setFinite_inter _ (IsBounded.smul₀ hs₁ y)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
      x y : Real
      hx : LT.lt 0 x
      hy : LE.le x y
      ⊢ HasSubset.Subset (Inter.inter (HSMul.hSMul x s) ↑(Submodule.span Int (Set.ra …
    -/
  · exact Set.inter_subset_inter_left _ <| hs₄ hx hy
    /-
      🎉 no goals
    -/


private theorem tendsto_card_div_pow₃ (hs₁ : IsBounded s)
    (hs₄ : ∀ ⦃x y : ℝ⦄, 0 < x → x ≤ y → x • s ⊆ y • s) :
    ∀ᶠ x : ℝ in atTop, (Nat.card ↑(s ∩ (⌊x⌋₊ : ℝ)⁻¹ • L) : ℝ) / x ^ card ι ≤
      (Nat.card ↑(s ∩ x⁻¹ • L) : ℝ) / x ^ card ι := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    ⊢ Filter.Eventually (fun x => LE.le (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HS …
  -/
  filter_upwards [eventually_ge_atTop 1] with x hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    x : Real
    hx : LE.le 1 x
    ⊢ LE.le (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑(Nat.flo …
  -/
  gcongr
  exact tendsto_card_div_pow₂ s hs₁ hs₄ (Nat.cast_pos.mpr (Nat.floor_pos.mpr hx))
    (Nat.floor_le (zero_le_one.trans hx))


private theorem tendsto_card_div_pow₄ (hs₁ : IsBounded s)
    (hs₄ : ∀ ⦃x y : ℝ⦄, 0 < x → x ≤ y → x • s ⊆ y • s) :
    ∀ᶠ x : ℝ in atTop, (Nat.card ↑(s ∩ x⁻¹ • L) : ℝ) / x ^ card ι ≤
      (Nat.card ↑(s ∩ (⌈x⌉₊ : ℝ)⁻¹ • L) : ℝ) / x ^ card ι := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    ⊢ Filter.Eventually (fun x => LE.le (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HS …
  -/
  filter_upwards [eventually_gt_atTop 0] with x hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    x : Real
    hx : LT.lt 0 x
    ⊢ LE.le (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv x) ↑(Subm …
  -/
  gcongr
  /-
    case h.hab.h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    x : Real
    hx : LT.lt 0 x
    ⊢ LE.le (Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv x) ↑(Submodule.span In …
  -/
  exact tendsto_card_div_pow₂ s hs₁ hs₄ hx (Nat.le_ceil _)
  /-
    🎉 no goals
  -/


private theorem tendsto_card_div_pow₅ :
    (fun x ↦ (Nat.card ↑(s ∩ (⌊x⌋₊ : ℝ)⁻¹ • L) : ℝ) / ⌊x⌋₊ ^ card ι * (⌊x⌋₊ / x) ^ card ι)
      =ᶠ[atTop] (fun x ↦ (Nat.card ↑(s ∩ (⌊x⌋₊ : ℝ)⁻¹ • L) : ℝ) / x ^ card ι) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    ⊢ Filter.atTop.EventuallyEq (fun x => HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter …
  -/
  filter_upwards [eventually_ge_atTop 1] with x hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : Real
    hx : LE.le 1 x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑ …
  -/
  have : 0 < ⌊x⌋₊ := Nat.floor_pos.mpr hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : Real
    hx : LE.le 1 x
    this : LT.lt 0 (Nat.floor x)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑ …
  -/
  rw [div_pow, mul_div, div_mul_cancel₀ _ (by positivity)]
  /-
    🎉 no goals
  -/


private theorem tendsto_card_div_pow₆ :
    (fun x ↦ (Nat.card ↑(s ∩ (⌈x⌉₊ : ℝ)⁻¹ • L) : ℝ) / ⌈x⌉₊ ^ card ι * (⌈x⌉₊ / x) ^ card ι)
          =ᶠ[atTop] (fun x ↦ (Nat.card ↑(s ∩ (⌈x⌉₊ : ℝ)⁻¹ • L) : ℝ) / x ^ card ι) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    ⊢ Filter.atTop.EventuallyEq (fun x => HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter …
  -/
  filter_upwards [eventually_ge_atTop 1] with x hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : Real
    hx : LE.le 1 x
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑ …
  -/
  have : 0 < ⌊x⌋₊ := Nat.floor_pos.mpr hx
  /-
    case h
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    x : Real
    hx : LE.le 1 x
    this : LT.lt 0 (Nat.floor x)
    ⊢ Eq (HMul.hMul (HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul (Inv.inv ↑ …
  -/
  rw [div_pow, mul_div, div_mul_cancel₀ _ (by positivity)]
  /-
    🎉 no goals
  -/


/-- A version of `tendsto_card_div_pow_atTop_volume` for a real variable. -/
theorem _root_.tendsto_card_div_pow_atTop_volume' (hs₁ : IsBounded s)
    (hs₂ : MeasurableSet s) (hs₃ : volume (frontier s) = 0)
    (hs₄ : ∀ ⦃x y : ℝ⦄, 0 < x → x ≤ y → x • s ⊆ y • s) :
    Tendsto (fun x : ℝ ↦ (Nat.card ↑(s ∩ x⁻¹ • L) : ℝ) / x ^ card ι)
      atTop (nhds (volume s).toReal) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : Set (ι → Real)
    hs₁ : Bornology.IsBounded s
    hs₂ : MeasurableSet s
    hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
    hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
  -/
  rw [show (volume s).toReal = (volume s).toReal * 1 ^ card ι by ring]
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' ?_ ?_
    (tendsto_card_div_pow₃ s hs₁ hs₄) (tendsto_card_div_pow₄ s hs₁ hs₄)
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
      ⊢ Filter.Tendsto (fun b => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
    -/
  · refine Tendsto.congr' (tendsto_card_div_pow₅ s) (Tendsto.mul ?_ (Tendsto.pow ?_ _))
      /-
        case refine_1.refine_1
        ι : Type u_1
        inst✝ : Fintype ι
        s : Set (ι → Real)
        hs₁ : Bornology.IsBounded s
        hs₂ : MeasurableSet s
        hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
        hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
        ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
      -/
    · exact Tendsto.comp (tendsto_card_div_pow_atTop_volume s hs₁ hs₂ hs₃) tendsto_nat_floor_atTop
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        ι : Type u_1
        inst✝ : Fintype ι
        s : Set (ι → Real)
        hs₁ : Bornology.IsBounded s
        hs₂ : MeasurableSet s
        hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
        hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
        ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.floor x)) x) Filter.atTop (nhds 1)
      -/
    · exact tendsto_nat_floor_div_atTop
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      s : Set (ι → Real)
      hs₁ : Bornology.IsBounded s
      hs₂ : MeasurableSet s
      hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
      hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
      ⊢ Filter.Tendsto (fun b => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
    -/
  · refine Tendsto.congr' (tendsto_card_div_pow₆ s) (Tendsto.mul ?_ (Tendsto.pow ?_ _))
      /-
        case refine_2.refine_1
        ι : Type u_1
        inst✝ : Fintype ι
        s : Set (ι → Real)
        hs₁ : Bornology.IsBounded s
        hs₂ : MeasurableSet s
        hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
        hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
        ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.card ↑(Inter.inter s (HSMul.hSMul  …
      -/
    · exact Tendsto.comp (tendsto_card_div_pow_atTop_volume s hs₁ hs₂ hs₃) tendsto_nat_ceil_atTop
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        ι : Type u_1
        inst✝ : Fintype ι
        s : Set (ι → Real)
        hs₁ : Bornology.IsBounded s
        hs₂ : MeasurableSet s
        hs₃ : Eq (MeasureTheory.MeasureSpace.volume (frontier s)) 0
        hs₄ : ∀ ⦃x y : Real⦄, LT.lt 0 x → LE.le x y → HasSubset.Subset (HSMul.hSMul x  …
        ⊢ Filter.Tendsto (fun x => HDiv.hDiv (↑(Nat.ceil x)) x) Filter.atTop (nhds 1)
      -/
    · exact tendsto_nat_ceil_div_atTop
      /-
        🎉 no goals
      -/


