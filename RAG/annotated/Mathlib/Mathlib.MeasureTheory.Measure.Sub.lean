/-- The measure `μ - ν` is defined to be the least measure `τ` such that `μ ≤ τ + ν`.
It is the equivalent of `(μ - ν) ⊔ 0` if `μ` and `ν` were signed measures.
Compare with `ENNReal.instSub`.
Specifically, note that if you have `α = {1,2}`, and `μ {1} = 2`, `μ {2} = 0`, and
`ν {2} = 2`, `ν {1} = 0`, then `(μ - ν) {1, 2} = 2`. However, if `μ ≤ ν`, and
`ν univ ≠ ∞`, then `(μ - ν) + ν = μ`. -/
noncomputable instance instSub {α : Type*} [MeasurableSpace α] : Sub (Measure α) :=
  ⟨fun μ ν => sInf { τ | μ ≤ τ + ν }⟩


theorem sub_def : μ - ν = sInf { d | μ ≤ d + ν } := rfl


theorem sub_le_of_le_add {d} (h : μ ≤ d + ν) : μ - ν ≤ d :=
  sInf_le h


theorem sub_eq_zero_of_le (h : μ ≤ ν) : μ - ν = 0 :=
                                                  /-
                                                    α : Type u_1
                                                    m : MeasurableSpace α
                                                    μ ν : MeasureTheory.Measure α
                                                    h : LE.le μ ν
                                                    ⊢ LE.le μ (HAdd.hAdd 0 ν)
                                                  -/
  nonpos_iff_eq_zero'.1 <| sub_le_of_le_add <| by rwa [zero_add]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem sub_le : μ - ν ≤ μ :=
  sub_le_of_le_add <| Measure.le_add_right le_rfl


@[simp]
theorem sub_top : μ - ⊤ = 0 :=
  sub_eq_zero_of_le le_top


@[simp]
theorem zero_sub : 0 - μ = 0 :=
  sub_eq_zero_of_le μ.zero_le


@[simp]
theorem sub_self : μ - μ = 0 :=
  sub_eq_zero_of_le le_rfl


/-- This application lemma only works in special circumstances. Given knowledge of
when `μ ≤ ν` and `ν ≤ μ`, a more general application lemma can be written. -/
theorem sub_apply [IsFiniteMeasure ν] (h₁ : MeasurableSet s) (h₂ : ν ≤ μ) :
    (μ - ν) s = μ s - ν s := by
  -- We begin by defining `measure_sub`, which will be equal to `(μ - ν)`.
  let measure_sub : Measure α := MeasureTheory.Measure.ofMeasurable
    (fun (t : Set α) (_ : MeasurableSet t) => μ t - ν t) (by simp)
    (fun g h_meas h_disj ↦ by
      simp only [measure_iUnion h_disj h_meas]
      rw [ENNReal.tsum_sub _ (h₂ <| g ·)]
      rw [← measure_iUnion h_disj h_meas]
      apply measure_ne_top)
  -- Now, we demonstrate `μ - ν = measure_sub`, and apply it.
  have h_measure_sub_add : ν + measure_sub = μ := by
    ext1 t h_t_measurable_set
    simp only [Pi.add_apply, coe_add]
    rw [MeasureTheory.Measure.ofMeasurable_apply _ h_t_measurable_set, add_comm,
      tsub_add_cancel_of_le (h₂ t)]
  have h_measure_sub_eq : μ - ν = measure_sub := by
    rw [MeasureTheory.Measure.sub_def]
    apply le_antisymm
    · apply sInf_le
      simp [le_refl, add_comm, h_measure_sub_add]
    apply le_sInf
    intro d h_d
    rw [← h_measure_sub_add, mem_setOf_eq, add_comm d] at h_d
    apply Measure.le_of_add_le_add_left h_d
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h₁ : MeasurableSet s
    h₂ : LE.le ν μ
    measure_sub : MeasureTheory.Measure α := MeasureTheory.Measure.ofMeasurable (f …
    h_measure_sub_add : Eq (HAdd.hAdd ν measure_sub) μ
    h_measure_sub_eq : Eq (HSub.hSub μ ν) measure_sub
    ⊢ Eq ((HSub.hSub μ ν) s) (HSub.hSub (μ s) (ν s))
  -/
  rw [h_measure_sub_eq]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h₁ : MeasurableSet s
    h₂ : LE.le ν μ
    measure_sub : MeasureTheory.Measure α := MeasureTheory.Measure.ofMeasurable (f …
    h_measure_sub_add : Eq (HAdd.hAdd ν measure_sub) μ
    h_measure_sub_eq : Eq (HSub.hSub μ ν) measure_sub
    ⊢ Eq (measure_sub s) (HSub.hSub (μ s) (ν s))
  -/
  apply Measure.ofMeasurable_apply _ h₁
  /-
    🎉 no goals
  -/


theorem sub_add_cancel_of_le [IsFiniteMeasure ν] (h₁ : ν ≤ μ) : μ - ν + ν = μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h₁ : LE.le ν μ
    ⊢ Eq (HAdd.hAdd (HSub.hSub μ ν) ν) μ
  -/
  ext1 s h_s_meas
  /-
    case h
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    h₁ : LE.le ν μ
    s : Set α
    h_s_meas : MeasurableSet s
    ⊢ Eq ((HAdd.hAdd (HSub.hSub μ ν) ν) s) (μ s)
  -/
  rw [add_apply, sub_apply h_s_meas h₁, tsub_add_cancel_of_le (h₁ s)]
  /-
    🎉 no goals
  -/


@[simp]
protected lemma add_sub_cancel [IsFiniteMeasure ν] : μ + ν - ν = μ := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    inst✝ : MeasureTheory.IsFiniteMeasure ν
    ⊢ Eq (HSub.hSub (HAdd.hAdd μ ν) ν) μ
  -/
  ext1 s hs
  rw [sub_apply hs (Measure.le_add_left (le_refl _)), add_apply,
    ENNReal.add_sub_cancel_right (measure_ne_top ν s)]


theorem restrict_sub_eq_restrict_sub_restrict (h_meas_s : MeasurableSet s) :
    (μ - ν).restrict s = μ.restrict s - ν.restrict s := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h_meas_s : MeasurableSet s
    ⊢ Eq ((HSub.hSub μ ν).restrict s) (HSub.hSub (μ.restrict s) (ν.restrict s))
  -/
  repeat rw [sub_def]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h_meas_s : MeasurableSet s
    ⊢ Eq ((InfSet.sInf (setOf fun d => LE.le μ (HAdd.hAdd d ν))).restrict s) (InfS …
  -/
  have h_nonempty : { d | μ ≤ d + ν }.Nonempty := ⟨μ, Measure.le_add_right le_rfl⟩
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h_meas_s : MeasurableSet s
    h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
    ⊢ Eq ((InfSet.sInf (setOf fun d => LE.le μ (HAdd.hAdd d ν))).restrict s) (InfS …
  -/
  rw [restrict_sInf_eq_sInf_restrict h_nonempty h_meas_s]
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h_meas_s : MeasurableSet s
    h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
    ⊢ Eq (InfSet.sInf (Set.image (fun μ => μ.restrict s) (setOf fun d => LE.le μ ( …
  -/
  apply le_antisymm
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ⊢ LE.le (InfSet.sInf (Set.image (fun μ => μ.restrict s) (setOf fun d => LE.le  …
    -/
  · refine sInf_le_sInf_of_forall_exists_le ?_
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ⊢ ∀ (x : MeasureTheory.Measure α), Membership.mem (setOf fun d => LE.le (μ.res …
    -/
    intro ν' h_ν'_in
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ν' : MeasureTheory.Measure α
      h_ν'_in : Membership.mem (setOf fun d => LE.le (μ.restrict s) (HAdd.hAdd d (ν. …
      ⊢ Exists fun y => And (Membership.mem (Set.image (fun μ => μ.restrict s) (setO …
    -/
    rw [mem_setOf_eq] at h_ν'_in
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ν' : MeasureTheory.Measure α
      h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
      ⊢ Exists fun y => And (Membership.mem (Set.image (fun μ => μ.restrict s) (setO …
    -/
    refine ⟨ν'.restrict s, ?_, restrict_le_self⟩
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ν' : MeasureTheory.Measure α
      h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
      ⊢ Membership.mem (Set.image (fun μ => μ.restrict s) (setOf fun d => LE.le μ (H …
    -/
    refine ⟨ν' + (⊤ : Measure α).restrict sᶜ, ?_, ?_⟩
      /-
        case a.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        ⊢ Membership.mem (setOf fun d => LE.le μ (HAdd.hAdd d ν)) (HAdd.hAdd ν' (Top.t …
      -/
    · rw [mem_setOf_eq, add_right_comm, Measure.le_iff]
      /-
        case a.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        ⊢ ∀ (s_1 : Set α), MeasurableSet s_1 → LE.le (μ s_1) ((HAdd.hAdd (HAdd.hAdd ν' …
      -/
      intro t h_meas_t
      /-
        case a.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        t : Set α
        h_meas_t : MeasurableSet t
        ⊢ LE.le (μ t) ((HAdd.hAdd (HAdd.hAdd ν' ν) (Top.top.restrict (HasCompl.compl s …
      -/
      repeat rw [← measure_inter_add_diff t h_meas_s]
      /-
        case a.refine_1
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        t : Set α
        h_meas_t : MeasurableSet t
        ⊢ LE.le (HAdd.hAdd (μ (Inter.inter t s)) (μ (SDiff.sdiff t s))) (HAdd.hAdd ((H …
      -/
      refine add_le_add ?_ ?_
        /-
          case a.refine_1.refine_1
          α : Type u_1
          m : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          h_meas_s : MeasurableSet s
          h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
          ν' : MeasureTheory.Measure α
          h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
          t : Set α
          h_meas_t : MeasurableSet t
          ⊢ LE.le (μ (Inter.inter t s)) ((HAdd.hAdd (HAdd.hAdd ν' ν) (Top.top.restrict ( …
        -/
      · rw [add_apply, add_apply]
        /-
          case a.refine_1.refine_1
          α : Type u_1
          m : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          h_meas_s : MeasurableSet s
          h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
          ν' : MeasureTheory.Measure α
          h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
          t : Set α
          h_meas_t : MeasurableSet t
          ⊢ LE.le (μ (Inter.inter t s)) (HAdd.hAdd (HAdd.hAdd (ν' (Inter.inter t s)) (ν  …
        -/
        apply le_add_right _
        rw [← restrict_eq_self μ inter_subset_right,
          ← restrict_eq_self ν inter_subset_right]
        /-
          α : Type u_1
          m : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          h_meas_s : MeasurableSet s
          h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
          ν' : MeasureTheory.Measure α
          h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
          t : Set α
          h_meas_t : MeasurableSet t
          ⊢ LE.le ((μ.restrict s) (Inter.inter t s)) (HAdd.hAdd (ν' (Inter.inter t s)) ( …
        -/
        apply h_ν'_in
        /-
          🎉 no goals
        -/
      · rw [add_apply, restrict_apply (h_meas_t.diff h_meas_s), diff_eq, inter_assoc, inter_self,
          ← add_apply]
        /-
          case a.refine_1.refine_2
          α : Type u_1
          m : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          h_meas_s : MeasurableSet s
          h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
          ν' : MeasureTheory.Measure α
          h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
          t : Set α
          h_meas_t : MeasurableSet t
          ⊢ LE.le (μ (Inter.inter t (HasCompl.compl s))) ((HAdd.hAdd (HAdd.hAdd ν' ν) To …
        -/
        have h_mu_le_add_top : μ ≤ ν' + ν + ⊤ := by simp only [add_top, le_top]
        /-
          case a.refine_1.refine_2
          α : Type u_1
          m : MeasurableSpace α
          μ ν : MeasureTheory.Measure α
          s : Set α
          h_meas_s : MeasurableSet s
          h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
          ν' : MeasureTheory.Measure α
          h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
          t : Set α
          h_meas_t : MeasurableSet t
          h_mu_le_add_top : LE.le μ (HAdd.hAdd (HAdd.hAdd ν' ν) Top.top)
          ⊢ LE.le (μ (Inter.inter t (HasCompl.compl s))) ((HAdd.hAdd (HAdd.hAdd ν' ν) To …
        -/
        exact Measure.le_iff'.1 h_mu_le_add_top _
        /-
          🎉 no goals
        -/
      /-
        case a.refine_2
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        ⊢ Eq ((fun μ => μ.restrict s) (HAdd.hAdd ν' (Top.top.restrict (HasCompl.compl  …
      -/
    · ext1 t h_meas_t
      /-
        case a.refine_2.h
        α : Type u_1
        m : MeasurableSpace α
        μ ν : MeasureTheory.Measure α
        s : Set α
        h_meas_s : MeasurableSet s
        h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
        ν' : MeasureTheory.Measure α
        h_ν'_in : LE.le (μ.restrict s) (HAdd.hAdd ν' (ν.restrict s))
        t : Set α
        h_meas_t : MeasurableSet t
        ⊢ Eq (((fun μ => μ.restrict s) (HAdd.hAdd ν' (Top.top.restrict (HasCompl.compl …
      -/
      simp [restrict_apply h_meas_t, restrict_apply (h_meas_t.inter h_meas_s), inter_assoc]
      /-
        🎉 no goals
      -/
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ⊢ LE.le (InfSet.sInf (setOf fun d => LE.le (μ.restrict s) (HAdd.hAdd d (ν.rest …
    -/
  · refine sInf_le_sInf_of_forall_exists_le ?_
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      ⊢ ∀ (x : MeasureTheory.Measure α), Membership.mem (Set.image (fun μ => μ.restr …
    -/
    refine forall_mem_image.2 fun t h_t_in => ⟨t.restrict s, ?_, le_rfl⟩
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      t : MeasureTheory.Measure α
      h_t_in : Membership.mem (setOf fun d => LE.le μ (HAdd.hAdd d ν)) t
      ⊢ Membership.mem (setOf fun d => LE.le (μ.restrict s) (HAdd.hAdd d (ν.restrict …
    -/
    rw [Set.mem_setOf_eq, ← restrict_add]
    /-
      case a
      α : Type u_1
      m : MeasurableSpace α
      μ ν : MeasureTheory.Measure α
      s : Set α
      h_meas_s : MeasurableSet s
      h_nonempty : (setOf fun d => LE.le μ (HAdd.hAdd d ν)).Nonempty
      t : MeasureTheory.Measure α
      h_t_in : Membership.mem (setOf fun d => LE.le μ (HAdd.hAdd d ν)) t
      ⊢ LE.le (μ.restrict s) ((HAdd.hAdd t ν).restrict s)
    -/
    exact restrict_mono Subset.rfl h_t_in
    /-
      🎉 no goals
    -/


theorem sub_apply_eq_zero_of_restrict_le_restrict (h_le : μ.restrict s ≤ ν.restrict s)
    (h_meas_s : MeasurableSet s) : (μ - ν) s = 0 := by
  /-
    α : Type u_1
    m : MeasurableSpace α
    μ ν : MeasureTheory.Measure α
    s : Set α
    h_le : LE.le (μ.restrict s) (ν.restrict s)
    h_meas_s : MeasurableSet s
    ⊢ Eq ((HSub.hSub μ ν) s) 0
  -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
  rw [← restrict_apply_self, restrict_sub_eq_restrict_sub_restrict, sub_eq_zero_of_le] <;> simp [*]
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/


instance isFiniteMeasure_sub [IsFiniteMeasure μ] : IsFiniteMeasure (μ - ν) :=
  isFiniteMeasure_of_le μ sub_le


