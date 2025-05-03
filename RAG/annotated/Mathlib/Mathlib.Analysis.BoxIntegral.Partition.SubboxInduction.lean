/-- Split a box in `ℝⁿ` into `2 ^ n` boxes by hyperplanes passing through its center. -/
def splitCenter (I : Box ι) : Prepartition I where
  boxes := Finset.univ.map (Box.splitCenterBoxEmb I)
                   /-
                     ι : Type u_1
                     inst✝ : Fintype ι
                     I✝ J I : BoxIntegral.Box ι
                     ⊢ ∀ (J : BoxIntegral.Box ι), Membership.mem (Finset.map I.splitCenterBoxEmb Fi …
                   -/
  le_of_mem' := by simp [I.splitCenterBox_le]
                   /-
                     🎉 no goals
                   -/
  pairwiseDisjoint := by
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      I✝ J I : BoxIntegral.Box ι
      ⊢ (↑(Finset.map I.splitCenterBoxEmb Finset.univ)).Pairwise (Function.onFun Dis …
    -/
    rw [Finset.coe_map, Finset.coe_univ, image_univ]
    /-
      ι : Type u_1
      inst✝ : Fintype ι
      I✝ J I : BoxIntegral.Box ι
      ⊢ (Set.range ⇑I.splitCenterBoxEmb).Pairwise (Function.onFun Disjoint BoxIntegr …
    -/
    rintro _ ⟨s, rfl⟩ _ ⟨t, rfl⟩ Hne
    /-
      case intro.intro
      ι : Type u_1
      inst✝ : Fintype ι
      I✝ J I : BoxIntegral.Box ι
      s t : Set ι
      Hne : Ne (I.splitCenterBoxEmb s) (I.splitCenterBoxEmb t)
      ⊢ Function.onFun Disjoint BoxIntegral.Box.toSet (I.splitCenterBoxEmb s) (I.spl …
    -/
    exact I.disjoint_splitCenterBox (mt (congr_arg _) Hne)
    /-
      🎉 no goals
    -/


@[simp]
                                                                                /-
                                                                                  ι : Type u_1
                                                                                  inst✝ : Fintype ι
                                                                                  I J : BoxIntegral.Box ι
                                                                                  ⊢ Iff (Membership.mem (BoxIntegral.Prepartition.splitCenter I) J) (Exists fun  …
                                                                                -/
theorem mem_splitCenter : J ∈ splitCenter I ↔ ∃ s, I.splitCenterBox s = J := by simp [splitCenter]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem isPartition_splitCenter (I : Box ι) : IsPartition (splitCenter I) := fun x hx => by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    x : ι → Real
    hx : Membership.mem I x
    ⊢ Exists fun J => And (Membership.mem (BoxIntegral.Prepartition.splitCenter I) …
  -/
  simp [hx]
  /-
    🎉 no goals
  -/


theorem upper_sub_lower_of_mem_splitCenter (h : J ∈ splitCenter I) (i : ι) :
    J.upper i - J.lower i = (I.upper i - I.lower i) / 2 :=
  let ⟨s, hs⟩ := mem_splitCenter.1 h
  hs ▸ I.upper_sub_lower_splitCenterBox s i


/-- Let `p` be a predicate on `Box ι`, let `I` be a box. Suppose that the following two properties
hold true.

* Consider a smaller box `J ≤ I`. The hyperplanes passing through the center of `J` split it into
  `2 ^ n` boxes. If `p` holds true on each of these boxes, then it true on `J`.
* For each `z` in the closed box `I.Icc` there exists a neighborhood `U` of `z` within `I.Icc` such
  that for every box `J ≤ I` such that `z ∈ J.Icc ⊆ U`, if `J` is homothetic to `I` with a
  coefficient of the form `1 / 2 ^ m`, then `p` is true on `J`.

Then `p I` is true. See also `BoxIntegral.Box.subbox_induction_on'` for a version using
`BoxIntegral.Box.splitCenterBox` instead of `BoxIntegral.Prepartition.splitCenter`. -/
@[elab_as_elim]
theorem subbox_induction_on {p : Box ι → Prop} (I : Box ι)
    (H_ind : ∀ J ≤ I, (∀ J' ∈ splitCenter J, p J') → p J)
    (H_nhds : ∀ z ∈ Box.Icc I, ∃ U ∈ 𝓝[Box.Icc I] z, ∀ J ≤ I, ∀ (m : ℕ),
      z ∈ Box.Icc J → Box.Icc J ⊆ U →
        (∀ i, J.upper i - J.lower i = (I.upper i - I.lower i) / 2 ^ m) → p J) :
    p I := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : BoxIntegral.Box ι → Prop
    I : BoxIntegral.Box ι
    H_ind : ∀ (J : BoxIntegral.Box ι), LE.le J I → (∀ (J' : BoxIntegral.Box ι), Me …
    H_nhds : ∀ (z : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) z → Exists f …
    ⊢ p I
  -/
  refine subbox_induction_on' I (fun J hle hs => H_ind J hle fun J' h' => ?_) H_nhds
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    p : BoxIntegral.Box ι → Prop
    I : BoxIntegral.Box ι
    H_ind : ∀ (J : BoxIntegral.Box ι), LE.le J I → (∀ (J' : BoxIntegral.Box ι), Me …
    H_nhds : ∀ (z : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) z → Exists f …
    J : BoxIntegral.Box ι
    hle : LE.le J I
    hs : ∀ (s : Set ι), p (J.splitCenterBox s)
    J' : BoxIntegral.Box ι
    h' : Membership.mem (BoxIntegral.Prepartition.splitCenter J) J'
    ⊢ p J'
  -/
  rcases mem_splitCenter.1 h' with ⟨s, rfl⟩
  /-
    case intro
    ι : Type u_1
    inst✝ : Fintype ι
    p : BoxIntegral.Box ι → Prop
    I : BoxIntegral.Box ι
    H_ind : ∀ (J : BoxIntegral.Box ι), LE.le J I → (∀ (J' : BoxIntegral.Box ι), Me …
    H_nhds : ∀ (z : ι → Real), Membership.mem (BoxIntegral.Box.Icc I) z → Exists f …
    J : BoxIntegral.Box ι
    hle : LE.le J I
    hs : ∀ (s : Set ι), p (J.splitCenterBox s)
    s : Set ι
    h' : Membership.mem (BoxIntegral.Prepartition.splitCenter J) (J.splitCenterBox …
    ⊢ p (J.splitCenterBox s)
  -/
  exact hs s
  /-
    🎉 no goals
  -/


/-- Given a box `I` in `ℝⁿ` and a function `r : ℝⁿ → (0, ∞)`, there exists a tagged partition `π` of
`I` such that

* `π` is a Henstock partition;
* `π` is subordinate to `r`;
* each box in `π` is homothetic to `I` with coefficient of the form `1 / 2 ^ m`.

This lemma implies that the Henstock filter is nontrivial, hence the Henstock integral is
well-defined. -/
theorem exists_taggedPartition_isHenstock_isSubordinate_homothetic (I : Box ι)
    (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    ∃ π : TaggedPrepartition I, π.IsPartition ∧ π.IsHenstock ∧ π.IsSubordinate r ∧
      (∀ J ∈ π, ∃ m : ℕ, ∀ i, (J : _).upper i - J.lower i = (I.upper i - I.lower i) / 2 ^ m) ∧
        π.distortion = I.distortion := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ Exists fun π => And π.IsPartition (And π.IsHenstock (And (π.IsSubordinate r) …
  -/
  refine subbox_induction_on I (fun J _ hJ => ?_) fun z _ => ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      hJ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      ⊢ Exists fun π => And π.IsPartition (And π.IsHenstock (And (π.IsSubordinate r) …
    -/
  · choose! πi hP hHen hr Hn _ using hJ
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      πi : (J' : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J'
      hP : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      hHen : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.sp …
      hr : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      Hn : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      a✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      ⊢ Exists fun π => And π.IsPartition (And π.IsHenstock (And (π.IsSubordinate r) …
    -/
    choose! n hn using Hn
    have hP : ((splitCenter J).biUnionTagged πi).IsPartition :=
      (isPartition_splitCenter _).biUnionTagged hP
    have hsub : ∀ J' ∈ (splitCenter J).biUnionTagged πi, ∃ n : ℕ, ∀ i,
        (J' : _).upper i - J'.lower i = (J.upper i - J.lower i) / 2 ^ n := by
      intro J' hJ'
      rcases (splitCenter J).mem_biUnionTagged.1 hJ' with ⟨J₁, h₁, h₂⟩
      refine ⟨n J₁ J' + 1, fun i => ?_⟩
      simp only [hn J₁ h₁ J' h₂, upper_sub_lower_of_mem_splitCenter h₁, pow_succ', div_div]
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      πi : (J' : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J'
      hP✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spl …
      hHen : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.sp …
      hr : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      a✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      n : BoxIntegral.Box ι → BoxIntegral.Box ι → Nat
      hn : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      hP : ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi).IsPartition
      hsub : ∀ (J' : BoxIntegral.Box ι), Membership.mem ((BoxIntegral.Prepartition.s …
      ⊢ Exists fun π => And π.IsPartition (And π.IsHenstock (And (π.IsSubordinate r) …
    -/
    refine ⟨_, hP, isHenstock_biUnionTagged.2 hHen, isSubordinate_biUnionTagged.2 hr, hsub, ?_⟩
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      πi : (J' : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J'
      hP✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spl …
      hHen : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.sp …
      hr : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      a✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      n : BoxIntegral.Box ι → BoxIntegral.Box ι → Nat
      hn : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      hP : ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi).IsPartition
      hsub : ∀ (J' : BoxIntegral.Box ι), Membership.mem ((BoxIntegral.Prepartition.s …
      ⊢ Eq ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi).distortion J. …
    -/
    refine TaggedPrepartition.distortion_of_const _ hP.nonempty_boxes fun J' h' => ?_
    /-
      case refine_1
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      πi : (J' : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J'
      hP✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spl …
      hHen : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.sp …
      hr : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      a✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      n : BoxIntegral.Box ι → BoxIntegral.Box ι → Nat
      hn : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      hP : ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi).IsPartition
      hsub : ∀ (J' : BoxIntegral.Box ι), Membership.mem ((BoxIntegral.Prepartition.s …
      J' : BoxIntegral.Box ι
      h' : Membership.mem ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi …
      ⊢ Eq J'.distortion J.distortion
    -/
    rcases hsub J' h' with ⟨n, hn⟩
    /-
      case refine_1.intro
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      J : BoxIntegral.Box ι
      x✝ : LE.le J I
      πi : (J' : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J'
      hP✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spl …
      hHen : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.sp …
      hr : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      a✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spli …
      n✝ : BoxIntegral.Box ι → BoxIntegral.Box ι → Nat
      hn✝ : ∀ (J' : BoxIntegral.Box ι), Membership.mem (BoxIntegral.Prepartition.spl …
      hP : ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi).IsPartition
      hsub : ∀ (J' : BoxIntegral.Box ι), Membership.mem ((BoxIntegral.Prepartition.s …
      J' : BoxIntegral.Box ι
      h' : Membership.mem ((BoxIntegral.Prepartition.splitCenter J).biUnionTagged πi …
      n : Nat
      hn : ∀ (i : ι), Eq (HSub.hSub (J'.upper i) (J'.lower i)) (HDiv.hDiv (HSub.hSub …
      ⊢ Eq J'.distortion J.distortion
    -/
    exact Box.distortion_eq_of_sub_eq_div hn
    /-
      🎉 no goals
    -/
  · refine ⟨Box.Icc I ∩ closedBall z (r z),
      inter_mem_nhdsWithin _ (closedBall_mem_nhds _ (r z).coe_prop), ?_⟩
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      z : ι → Real
      x✝ : Membership.mem (BoxIntegral.Box.Icc I) z
      ⊢ ∀ (J : BoxIntegral.Box ι), LE.le J I → ∀ (m : Nat), Membership.mem (BoxInteg …
    -/
    intro J _ n Hmem HIcc Hsub
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      z : ι → Real
      x✝ : Membership.mem (BoxIntegral.Box.Icc I) z
      J : BoxIntegral.Box ι
      a✝ : LE.le J I
      n : Nat
      Hmem : Membership.mem (BoxIntegral.Box.Icc J) z
      HIcc : HasSubset.Subset (BoxIntegral.Box.Icc J) (Inter.inter (BoxIntegral.Box. …
      Hsub : ∀ (i : ι), Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv (HSub.hSub …
      ⊢ Exists fun π => And π.IsPartition (And π.IsHenstock (And (π.IsSubordinate r) …
    -/
    rw [Set.subset_inter_iff] at HIcc
    refine ⟨single _ _ le_rfl _ Hmem, isPartition_single _, isHenstock_single _,
      (isSubordinate_single _ _).2 HIcc.2, ?_, distortion_single _ _⟩
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      z : ι → Real
      x✝ : Membership.mem (BoxIntegral.Box.Icc I) z
      J : BoxIntegral.Box ι
      a✝ : LE.le J I
      n : Nat
      Hmem : Membership.mem (BoxIntegral.Box.Icc J) z
      HIcc : And (HasSubset.Subset (BoxIntegral.Box.Icc J) (BoxIntegral.Box.Icc I))  …
      Hsub : ∀ (i : ι), Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv (HSub.hSub …
      ⊢ ∀ (J_1 : BoxIntegral.Box ι), Membership.mem (BoxIntegral.TaggedPrepartition. …
    -/
    simp only [TaggedPrepartition.mem_single, forall_eq]
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      z : ι → Real
      x✝ : Membership.mem (BoxIntegral.Box.Icc I) z
      J : BoxIntegral.Box ι
      a✝ : LE.le J I
      n : Nat
      Hmem : Membership.mem (BoxIntegral.Box.Icc J) z
      HIcc : And (HasSubset.Subset (BoxIntegral.Box.Icc J) (BoxIntegral.Box.Icc I))  …
      Hsub : ∀ (i : ι), Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv (HSub.hSub …
      ⊢ Exists fun m => ∀ (i : ι), Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv …
    -/
    refine ⟨0, fun i => ?_⟩
    /-
      case refine_2
      ι : Type u_1
      inst✝ : Fintype ι
      I : BoxIntegral.Box ι
      r : (ι → Real) → ↑(Set.Ioi 0)
      z : ι → Real
      x✝ : Membership.mem (BoxIntegral.Box.Icc I) z
      J : BoxIntegral.Box ι
      a✝ : LE.le J I
      n : Nat
      Hmem : Membership.mem (BoxIntegral.Box.Icc J) z
      HIcc : And (HasSubset.Subset (BoxIntegral.Box.Icc J) (BoxIntegral.Box.Icc I))  …
      Hsub : ∀ (i : ι), Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv (HSub.hSub …
      i : ι
      ⊢ Eq (HSub.hSub (J.upper i) (J.lower i)) (HDiv.hDiv (HSub.hSub (J.upper i) (J. …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- Given a box `I` in `ℝⁿ`, a function `r : ℝⁿ → (0, ∞)`, and a prepartition `π` of `I`, there
exists a tagged prepartition `π'` of `I` such that

* each box of `π'` is included in some box of `π`;
* `π'` is a Henstock partition;
* `π'` is subordinate to `r`;
* `π'` covers exactly the same part of `I` as `π`;
* the distortion of `π'` is equal to the distortion of `π`.
-/
theorem exists_tagged_le_isHenstock_isSubordinate_iUnion_eq {I : Box ι} (r : (ι → ℝ) → Ioi (0 : ℝ))
    (π : Prepartition I) :
    ∃ π' : TaggedPrepartition I, π'.toPrepartition ≤ π ∧ π'.IsHenstock ∧ π'.IsSubordinate r ∧
      π'.distortion = π.distortion ∧ π'.iUnion = π.iUnion := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.Prepartition I
    ⊢ Exists fun π' => And (LE.le π'.toPrepartition π) (And π'.IsHenstock (And (π' …
  -/
  have := fun J => Box.exists_taggedPartition_isHenstock_isSubordinate_homothetic J r
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.Prepartition I
    this : ∀ (J : BoxIntegral.Box ι), Exists fun π => And π.IsPartition (And π.IsH …
    ⊢ Exists fun π' => And (LE.le π'.toPrepartition π) (And π'.IsHenstock (And (π' …
  -/
  choose! πi πip πiH πir _ πid using this
  refine ⟨π.biUnionTagged πi, biUnion_le _ _, isHenstock_biUnionTagged.2 fun J _ => πiH J,
    isSubordinate_biUnionTagged.2 fun J _ => πir J, ?_, π.iUnion_biUnion_partition fun J _ => πip J⟩
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.Prepartition I
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    πip : ∀ (J : BoxIntegral.Box ι), (πi J).IsPartition
    πiH : ∀ (J : BoxIntegral.Box ι), (πi J).IsHenstock
    πir : ∀ (J : BoxIntegral.Box ι), (πi J).IsSubordinate r
    h✝ : ∀ (J J_1 : BoxIntegral.Box ι), Membership.mem (πi J) J_1 → Exists fun m = …
    πid : ∀ (J : BoxIntegral.Box ι), Eq (πi J).distortion J.distortion
    ⊢ Eq (π.biUnionTagged πi).distortion π.distortion
  -/
  rw [distortion_biUnionTagged]
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    r : (ι → Real) → ↑(Set.Ioi 0)
    π : BoxIntegral.Prepartition I
    πi : (J : BoxIntegral.Box ι) → BoxIntegral.TaggedPrepartition J
    πip : ∀ (J : BoxIntegral.Box ι), (πi J).IsPartition
    πiH : ∀ (J : BoxIntegral.Box ι), (πi J).IsHenstock
    πir : ∀ (J : BoxIntegral.Box ι), (πi J).IsSubordinate r
    h✝ : ∀ (J J_1 : BoxIntegral.Box ι), Membership.mem (πi J) J_1 → Exists fun m = …
    πid : ∀ (J : BoxIntegral.Box ι), Eq (πi J).distortion J.distortion
    ⊢ Eq (π.boxes.sup fun J => (πi J).distortion) π.distortion
  -/
  exact sup_congr rfl fun J _ => πid J
  /-
    🎉 no goals
  -/


/-- Given a prepartition `π` of a box `I` and a function `r : ℝⁿ → (0, ∞)`, `π.toSubordinate r`
is a tagged partition `π'` such that

* each box of `π'` is included in some box of `π`;
* `π'` is a Henstock partition;
* `π'` is subordinate to `r`;
* `π'` covers exactly the same part of `I` as `π`;
* the distortion of `π'` is equal to the distortion of `π`.
-/
def toSubordinate (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) : TaggedPrepartition I :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose


theorem toSubordinate_toPrepartition_le (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π.toSubordinate r).toPrepartition ≤ π :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose_spec.1


theorem isHenstock_toSubordinate (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π.toSubordinate r).IsHenstock :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose_spec.2.1


theorem isSubordinate_toSubordinate (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π.toSubordinate r).IsSubordinate r :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose_spec.2.2.1


@[simp]
theorem distortion_toSubordinate (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π.toSubordinate r).distortion = π.distortion :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose_spec.2.2.2.1


@[simp]
theorem iUnion_toSubordinate (π : Prepartition I) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π.toSubordinate r).iUnion = π.iUnion :=
  (π.exists_tagged_le_isHenstock_isSubordinate_iUnion_eq r).choose_spec.2.2.2.2


/-- Given a tagged prepartition `π₁`, a prepartition `π₂` that covers exactly `I \ π₁.iUnion`, and
a function `r : ℝⁿ → (0, ∞)`, returns the union of `π₁` and `π₂.toSubordinate r`. This partition
`π` has the following properties:

* `π` is a partition, i.e. it covers the whole `I`;
* `π₁.boxes ⊆ π.boxes`;
* `π.tag J = π₁.tag J` whenever `J ∈ π₁`;
* `π` is Henstock outside of `π₁`: `π.tag J ∈ J.Icc` whenever `J ∈ π`, `J ∉ π₁`;
* `π` is subordinate to `r` outside of `π₁`;
* the distortion of `π` is equal to the maximum of the distortions of `π₁` and `π₂`.
-/
def unionComplToSubordinate (π₁ : TaggedPrepartition I) (π₂ : Prepartition I)
    (hU : π₂.iUnion = ↑I \ π₁.iUnion) (r : (ι → ℝ) → Ioi (0 : ℝ)) : TaggedPrepartition I :=
  π₁.disjUnion (π₂.toSubordinate r)
    (((π₂.iUnion_toSubordinate r).trans hU).symm ▸ disjoint_sdiff_self_right)


theorem isPartition_unionComplToSubordinate (π₁ : TaggedPrepartition I) (π₂ : Prepartition I)
    (hU : π₂.iUnion = ↑I \ π₁.iUnion) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    IsPartition (π₁.unionComplToSubordinate π₂ hU r) :=
  Prepartition.isPartitionDisjUnionOfEqDiff ((π₂.iUnion_toSubordinate r).trans hU)


open scoped Classical in
@[simp]
theorem unionComplToSubordinate_boxes (π₁ : TaggedPrepartition I) (π₂ : Prepartition I)
    (hU : π₂.iUnion = ↑I \ π₁.iUnion) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π₁.unionComplToSubordinate π₂ hU r).boxes = π₁.boxes ∪ (π₂.toSubordinate r).boxes := rfl


@[simp]
theorem iUnion_unionComplToSubordinate_boxes (π₁ : TaggedPrepartition I) (π₂ : Prepartition I)
    (hU : π₂.iUnion = ↑I \ π₁.iUnion) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π₁.unionComplToSubordinate π₂ hU r).iUnion = I :=
  (isPartition_unionComplToSubordinate _ _ _ _).iUnion_eq


@[simp]
theorem distortion_unionComplToSubordinate (π₁ : TaggedPrepartition I) (π₂ : Prepartition I)
    (hU : π₂.iUnion = ↑I \ π₁.iUnion) (r : (ι → ℝ) → Ioi (0 : ℝ)) :
    (π₁.unionComplToSubordinate π₂ hU r).distortion = max π₁.distortion π₂.distortion := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    I : BoxIntegral.Box ι
    π₁ : BoxIntegral.TaggedPrepartition I
    π₂ : BoxIntegral.Prepartition I
    hU : Eq π₂.iUnion (SDiff.sdiff (↑I) π₁.iUnion)
    r : (ι → Real) → ↑(Set.Ioi 0)
    ⊢ Eq (π₁.unionComplToSubordinate π₂ hU r).distortion (Max.max π₁.distortion π₂ …
  -/
  simp [unionComplToSubordinate]
  /-
    🎉 no goals
  -/


