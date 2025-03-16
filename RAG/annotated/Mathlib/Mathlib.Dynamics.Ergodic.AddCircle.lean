/-- If a null-measurable subset of the circle is almost invariant under rotation by a family of
rational angles with denominators tending to infinity, then it must be almost empty or almost full.
-/
theorem ae_empty_or_univ_of_forall_vadd_ae_eq_self {s : Set <| AddCircle T}
    (hs : NullMeasurableSet s volume) {ι : Type*} {l : Filter ι} [l.NeBot] {u : ι → AddCircle T}
    (hu₁ : ∀ i, (u i +ᵥ s : Set _) =ᵐ[volume] s) (hu₂ : Tendsto (addOrderOf ∘ u) l atTop) :
    s =ᵐ[volume] (∅ : Set <| AddCircle T) ∨ s =ᵐ[volume] univ := by
  /- Sketch of proof:
    Assume `T = 1` for simplicity and let `μ` be the Haar measure. We may assume `s` has positive
    measure since otherwise there is nothing to prove. In this case, by Lebesgue's density theorem,
    there exists a point `d` of positive density. Let `Iⱼ` be the sequence of closed balls about `d`
    of diameter `1 / nⱼ` where `nⱼ` is the additive order of `uⱼ`. Since `d` has positive density we
    must have `μ (s ∩ Iⱼ) / μ Iⱼ → 1` along `l`. However since `s` is invariant under the action of
    `uⱼ` and since `Iⱼ` is a fundamental domain for this action, we must have
    `μ (s ∩ Iⱼ) = nⱼ * μ s = (μ Iⱼ) * μ s`. We thus have `μ s → 1` and thus `μ s = 1`. -/
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    hs : MeasureTheory.NullMeasurableSet s MeasureTheory.MeasureSpace.volume
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    hu₁ : ∀ (i : ι), (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).Eventual …
    hu₂ : Filter.Tendsto (Function.comp addOrderOf u) l Filter.atTop
    ⊢ Or ((MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq s Empt …
  -/
  set μ := (volume : Measure <| AddCircle T)
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    hu₂ : Filter.Tendsto (Function.comp addOrderOf u) l Filter.atTop
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  set n : ι → ℕ := addOrderOf ∘ u
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  have hT₀ : 0 < T := hT.out
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  have hT₁ : ENNReal.ofReal T ≠ 0 := by simpa
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    ⊢ Or ((MeasureTheory.ae μ).EventuallyEq s EmptyCollection.emptyCollection) ((M …
  -/
  rw [ae_eq_empty, ae_eq_univ_iff_measure_eq hs, AddCircle.measure_univ]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    ⊢ Or (Eq (μ s) 0) (Eq (μ s) (ENNReal.ofReal T))
  -/
  rcases eq_or_ne (μ s) 0 with h | h; · exact Or.inl h
                                        /-
                                          🎉 no goals
                                        -/
  /-
    case inr
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    ⊢ Or (Eq (μ s) 0) (Eq (μ s) (ENNReal.ofReal T))
  -/
  right
  obtain ⟨d, -, hd⟩ : ∃ d, d ∈ s ∧ ∀ {ι'} {l : Filter ι'} (w : ι' → AddCircle T) (δ : ι' → ℝ),
    Tendsto δ l (𝓝[>] 0) → (∀ᶠ j in l, d ∈ closedBall (w j) (1 * δ j)) →
      Tendsto (fun j => μ (s ∩ closedBall (w j) (δ j)) / μ (closedBall (w j) (δ j))) l (𝓝 1) :=
    exists_mem_of_measure_ne_zero_of_ae h
      (IsUnifLocDoublingMeasure.ae_tendsto_measure_inter_div μ s 1)
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    hd : ∀ {ι' : Type ?u.6444} {l : Filter ι'} (w : ι' → AddCircle T) (δ : ι' → Re …
    ⊢ Eq (μ s) (ENNReal.ofReal T)
  -/
  let I : ι → Set (AddCircle T) := fun j => closedBall d (T / (2 * ↑(n j)))
  replace hd : Tendsto (fun j => μ (s ∩ I j) / μ (I j)) l (𝓝 1) := by
    let δ : ι → ℝ := fun j => T / (2 * ↑(n j))
    have hδ₀ : ∀ᶠ j in l, 0 < δ j :=
      (hu₂.eventually_gt_atTop 0).mono fun j hj => div_pos hT₀ <| by positivity
    have hδ₁ : Tendsto δ l (𝓝[>] 0) := by
      refine tendsto_nhdsWithin_iff.mpr ⟨?_, hδ₀⟩
      replace hu₂ : Tendsto (fun j => T⁻¹ * 2 * n j) l atTop :=
        (tendsto_natCast_atTop_iff.mpr hu₂).const_mul_atTop (by positivity : 0 < T⁻¹ * 2)
      convert hu₂.inv_tendsto_atTop
      ext j
      simp only [δ, Pi.inv_apply, mul_inv_rev, inv_inv, div_eq_inv_mul, ← mul_assoc]
    have hw : ∀ᶠ j in l, d ∈ closedBall d (1 * δ j) := hδ₀.mono fun j hj => by
      simp only [comp_apply, one_mul, mem_closedBall, dist_self]
      apply hj.le
    exact hd _ δ hδ₁ hw
  suffices ∀ᶠ j in l, μ (s ∩ I j) / μ (I j) = μ s / ENNReal.ofReal T by
    replace hd := hd.congr' this
    rwa [tendsto_const_nhds_iff, ENNReal.div_eq_one_iff hT₁ ENNReal.ofReal_ne_top] at hd
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    ⊢ Filter.Eventually (fun j => Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j) …
  -/
  refine (hu₂.eventually_gt_atTop 0).mono fun j hj => ?_
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    j : ι
    hj : LT.lt 0 (n j)
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) (HDiv.hDiv (μ s) (ENNReal …
  -/
  have : addOrderOf (u j) = n j := rfl
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    j : ι
    hj : LT.lt 0 (n j)
    this : Eq (addOrderOf (u j)) (n j)
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) (HDiv.hDiv (μ s) (ENNReal …
  -/
  have huj : IsOfFinAddOrder (u j) := addOrderOf_pos_iff.mp hj
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    j : ι
    hj : LT.lt 0 (n j)
    this : Eq (addOrderOf (u j)) (n j)
    huj : IsOfFinAddOrder (u j)
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) (HDiv.hDiv (μ s) (ENNReal …
  -/
  have huj' : 1 ≤ (↑(n j) : ℝ) := by norm_cast
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    j : ι
    hj : LT.lt 0 (n j)
    this : Eq (addOrderOf (u j)) (n j)
    huj : IsOfFinAddOrder (u j)
    huj' : LE.le 1 ↑(n j)
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) (HDiv.hDiv (μ s) (ENNReal …
  -/
  have hI₀ : μ (I j) ≠ 0 := (measure_closedBall_pos _ d <| by positivity).ne.symm
  /-
    case inr.h.intro.intro
    T : Real
    hT : Fact (LT.lt 0 T)
    s : Set (AddCircle T)
    ι : Type u_1
    l : Filter ι
    inst✝ : l.NeBot
    u : ι → AddCircle T
    μ : MeasureTheory.Measure (AddCircle T) := MeasureTheory.MeasureSpace.volume
    hs : MeasureTheory.NullMeasurableSet s μ
    hu₁ : ∀ (i : ι), (MeasureTheory.ae μ).EventuallyEq (HVAdd.hVAdd (u i) s) s
    n : ι → Nat := Function.comp addOrderOf u
    hu₂ : Filter.Tendsto n l Filter.atTop
    hT₀ : LT.lt 0 T
    hT₁ : Ne (ENNReal.ofReal T) 0
    h : Ne (μ s) 0
    d : AddCircle T
    I : ι → Set (AddCircle T) := fun j => Metric.closedBall d (HDiv.hDiv T (HMul.h …
    hd : Filter.Tendsto (fun j => HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) l …
    j : ι
    hj : LT.lt 0 (n j)
    this : Eq (addOrderOf (u j)) (n j)
    huj : IsOfFinAddOrder (u j)
    huj' : LE.le 1 ↑(n j)
    hI₀ : Ne (μ (I j)) 0
    ⊢ Eq (HDiv.hDiv (μ (Inter.inter s (I j))) (μ (I j))) (HDiv.hDiv (μ s) (ENNReal …
  -/
  have hI₁ : μ (I j) ≠ ⊤ := measure_ne_top _ _
  have hI₂ : μ (I j) * ↑(n j) = ENNReal.ofReal T := by
    rw [volume_closedBall, mul_div, mul_div_mul_left T _ two_ne_zero,
      min_eq_right (div_le_self hT₀.le huj'), mul_comm, ← nsmul_eq_mul, ← ENNReal.ofReal_nsmul,
      nsmul_eq_mul, mul_div_cancel₀]
    exact Nat.cast_ne_zero.mpr hj.ne'
  rw [ENNReal.div_eq_div_iff hT₁ ENNReal.ofReal_ne_top hI₀ hI₁,
    volume_of_add_preimage_eq s _ (u j) d huj (hu₁ j) closedBall_ae_eq_ball, nsmul_eq_mul, ←
    mul_assoc, this, hI₂]


                                               /-
                                                 T : Real
                                                 hT : Fact (LT.lt 0 T)
                                                 n : Int
                                                 hn : LT.lt 1 (abs n)
                                                 ⊢ MeasureTheory.Measure (AddCircle T)
                                               -/
theorem ergodic_zsmul {n : ℤ} (hn : 1 < |n|) : Ergodic fun y : AddCircle T => n • y :=
                                               /-
                                                 🎉 no goals
                                               -/
  { measurePreserving_zsmul volume (abs_pos.mp <| lt_trans zero_lt_one hn) with
    aeconst_set := fun s hs hs' => by
      /-
        T : Real
        hT : Fact (LT.lt 0 T)
        n : Int
        hn : LT.lt 1 (abs n)
        s : Set (AddCircle T)
        hs : MeasurableSet s
        hs' : Eq (Set.preimage (fun y => HSMul.hSMul n y) s) s
        ⊢ Filter.EventuallyConst s (MeasureTheory.ae MeasureTheory.MeasureSpace.volume)
      -/
      let u : ℕ → AddCircle T := fun j => ↑((↑1 : ℝ) / ↑(n.natAbs ^ j) * T)
      /-
        T : Real
        hT : Fact (LT.lt 0 T)
        n : Int
        hn : LT.lt 1 (abs n)
        s : Set (AddCircle T)
        hs : MeasurableSet s
        hs' : Eq (Set.preimage (fun y => HSMul.hSMul n y) s) s
        u : Nat → AddCircle T := fun j => ↑(HMul.hMul (HDiv.hDiv 1 ↑(HPow.hPow n.natAb …
        ⊢ Filter.EventuallyConst s (MeasureTheory.ae MeasureTheory.MeasureSpace.volume)
      -/
      replace hn : 1 < n.natAbs := by rwa [Int.abs_eq_natAbs, Nat.one_lt_cast] at hn
      have hu₀ : ∀ j, addOrderOf (u j) = n.natAbs ^ j := fun j => by
        convert addOrderOf_div_of_gcd_eq_one (p := T) (m := 1)
          (pow_pos (pos_of_gt hn) j) (gcd_one_left _)
        norm_cast
      have hnu : ∀ j, n ^ j • u j = 0 := fun j => by
        rw [← addOrderOf_dvd_iff_zsmul_eq_zero, hu₀, Int.natCast_pow, Int.natCast_natAbs, ← abs_pow,
          abs_dvd]
      have hu₁ : ∀ j, (u j +ᵥ s : Set _) =ᵐ[volume] s := fun j => by
        rw [vadd_eq_self_of_preimage_zsmul_eq_self hs' (hnu j)]
      have hu₂ : Tendsto (fun j => addOrderOf <| u j) atTop atTop := by
        simp_rw [hu₀]; exact Nat.tendsto_pow_atTop_atTop_of_one_lt hn
      /-
        T : Real
        hT : Fact (LT.lt 0 T)
        n : Int
        hn✝ : LT.lt 1 (abs n)
        s : Set (AddCircle T)
        hs : MeasurableSet s
        hs' : Eq (Set.preimage (fun y => HSMul.hSMul n y) s) s
        u : Nat → AddCircle T := fun j => ↑(HMul.hMul (HDiv.hDiv 1 ↑(HPow.hPow n.natAb …
        hn : LT.lt 1 n.natAbs
        hu₀ : ∀ (j : Nat), Eq (addOrderOf (u j)) (HPow.hPow n.natAbs j)
        hnu : ∀ (j : Nat), Eq (HSMul.hSMul (HPow.hPow n j) (u j)) 0
        hu₁ : ∀ (j : Nat), (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).Eventu …
        hu₂ : Filter.Tendsto (fun j => addOrderOf (u j)) Filter.atTop Filter.atTop
        ⊢ Filter.EventuallyConst s (MeasureTheory.ae MeasureTheory.MeasureSpace.volume)
      -/
      rw [eventuallyConst_set']
      /-
        T : Real
        hT : Fact (LT.lt 0 T)
        n : Int
        hn✝ : LT.lt 1 (abs n)
        s : Set (AddCircle T)
        hs : MeasurableSet s
        hs' : Eq (Set.preimage (fun y => HSMul.hSMul n y) s) s
        u : Nat → AddCircle T := fun j => ↑(HMul.hMul (HDiv.hDiv 1 ↑(HPow.hPow n.natAb …
        hn : LT.lt 1 n.natAbs
        hu₀ : ∀ (j : Nat), Eq (addOrderOf (u j)) (HPow.hPow n.natAbs j)
        hnu : ∀ (j : Nat), Eq (HSMul.hSMul (HPow.hPow n j) (u j)) 0
        hu₁ : ∀ (j : Nat), (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).Eventu …
        hu₂ : Filter.Tendsto (fun j => addOrderOf (u j)) Filter.atTop Filter.atTop
        ⊢ Or ((MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq s Empt …
      -/
      exact ae_empty_or_univ_of_forall_vadd_ae_eq_self hs.nullMeasurableSet hu₁ hu₂ }
      /-
        🎉 no goals
      -/


                                             /-
                                               T : Real
                                               hT : Fact (LT.lt 0 T)
                                               n : Nat
                                               hn : LT.lt 1 n
                                               ⊢ MeasureTheory.Measure (AddCircle T)
                                             -/
theorem ergodic_nsmul {n : ℕ} (hn : 1 < n) : Ergodic fun y : AddCircle T => n • y :=
                                             /-
                                               🎉 no goals
                                             -/
                    /-
                      T : Real
                      hT : Fact (LT.lt 0 T)
                      n : Nat
                      hn : LT.lt 1 n
                      ⊢ LT.lt 1 (abs ↑n)
                    -/
  ergodic_zsmul (by simp [hn] : 1 < |(n : ℤ)|)
                    /-
                      🎉 no goals
                    -/


                                                                    /-
                                                                      T : Real
                                                                      hT : Fact (LT.lt 0 T)
                                                                      x : AddCircle T
                                                                      n : Int
                                                                      h : LT.lt 1 (abs n)
                                                                      ⊢ MeasureTheory.Measure (AddCircle T)
                                                                    -/
theorem ergodic_zsmul_add (x : AddCircle T) {n : ℤ} (h : 1 < |n|) : Ergodic fun y => n • y + x := by
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    n : Int
    h : LT.lt 1 (abs n)
    ⊢ Ergodic (fun y => HAdd.hAdd (HSMul.hSMul n y) x) MeasureTheory.MeasureSpace. …
  -/
  set f : AddCircle T → AddCircle T := fun y => n • y + x
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    n : Int
    h : LT.lt 1 (abs n)
    f : AddCircle T → AddCircle T := fun y => HAdd.hAdd (HSMul.hSMul n y) x
    ⊢ Ergodic f MeasureTheory.MeasureSpace.volume
  -/
  let e : AddCircle T ≃ᵐ AddCircle T := MeasurableEquiv.addLeft (DivisibleBy.div x <| n - 1)
  have he : MeasurePreserving e volume volume :=
    measurePreserving_add_left volume (DivisibleBy.div x <| n - 1)
  suffices e ∘ f ∘ e.symm = fun y => n • y by
    rw [← he.ergodic_conjugate_iff, this]; exact ergodic_zsmul h
  replace h : n - 1 ≠ 0 := by
    rw [← abs_one] at h; rw [sub_ne_zero]; exact ne_of_apply_ne _ (ne_of_gt h)
  have hnx : n • DivisibleBy.div x (n - 1) = x + DivisibleBy.div x (n - 1) := by
    conv_rhs => congr; rw [← DivisibleBy.div_cancel x h]
    rw [sub_smul, one_smul, sub_add_cancel]
  /-
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    n : Int
    f : AddCircle T → AddCircle T := fun y => HAdd.hAdd (HSMul.hSMul n y) x
    e : MeasurableEquiv (AddCircle T) (AddCircle T) := MeasurableEquiv.addLeft (Di …
    he : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume Me …
    h : Ne (HSub.hSub n 1) 0
    hnx : Eq (HSMul.hSMul n (DivisibleBy.div x (HSub.hSub n 1))) (HAdd.hAdd x (Div …
    ⊢ Eq (Function.comp (⇑e) (Function.comp f ⇑e.symm)) fun y => HSMul.hSMul n y
  -/
  ext y
  simp only [f, e, hnx, MeasurableEquiv.coe_addLeft, MeasurableEquiv.symm_addLeft, comp_apply,
    smul_add, zsmul_neg', neg_smul, neg_add_rev]
  /-
    case h
    T : Real
    hT : Fact (LT.lt 0 T)
    x : AddCircle T
    n : Int
    f : AddCircle T → AddCircle T := fun y => HAdd.hAdd (HSMul.hSMul n y) x
    e : MeasurableEquiv (AddCircle T) (AddCircle T) := MeasurableEquiv.addLeft (Di …
    he : MeasureTheory.MeasurePreserving (⇑e) MeasureTheory.MeasureSpace.volume Me …
    h : Ne (HSub.hSub n 1) 0
    hnx : Eq (HSMul.hSMul n (DivisibleBy.div x (HSub.hSub n 1))) (HAdd.hAdd x (Div …
    y : AddCircle T
    ⊢ Eq (HAdd.hAdd (DivisibleBy.div x (HSub.hSub n 1)) (HAdd.hAdd (HAdd.hAdd (HAd …
  -/
  /-
    🎉 no goals
  -/
  abel
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    T : Real
                                                                    hT : Fact (LT.lt 0 T)
                                                                    x : AddCircle T
                                                                    n : Nat
                                                                    h : LT.lt 1 n
                                                                    ⊢ MeasureTheory.Measure (AddCircle T)
                                                                  -/
theorem ergodic_nsmul_add (x : AddCircle T) {n : ℕ} (h : 1 < n) : Ergodic fun y => n • y + x :=
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                          /-
                            T : Real
                            hT : Fact (LT.lt 0 T)
                            x : AddCircle T
                            n : Nat
                            h : LT.lt 1 n
                            ⊢ LT.lt 1 (abs ↑n)
                          -/
  ergodic_zsmul_add x (by simp [h] : 1 < |(n : ℤ)|)
                          /-
                            🎉 no goals
                          -/


