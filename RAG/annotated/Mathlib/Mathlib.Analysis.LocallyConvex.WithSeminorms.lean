/-- An abbreviation for indexed families of seminorms. This is mainly to allow for dot-notation. -/
abbrev SeminormFamily :=
  ι → Seminorm 𝕜 E


/-- The sets of a filter basis for the neighborhood filter of 0. -/
def basisSets (p : SeminormFamily 𝕜 E ι) : Set (Set E) :=
  ⋃ (s : Finset ι) (r) (_ : 0 < r), singleton (ball (s.sup p) (0 : E) r)


theorem basisSets_iff {U : Set E} :
    U ∈ p.basisSets ↔ ∃ (i : Finset ι) (r : ℝ), 0 < r ∧ U = ball (i.sup p) 0 r := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    ⊢ Iff (Membership.mem p.basisSets U) (Exists fun i => Exists fun r => And (LT. …
  -/
  simp only [basisSets, mem_iUnion, exists_prop, mem_singleton_iff]
  /-
    🎉 no goals
  -/


theorem basisSets_mem (i : Finset ι) {r : ℝ} (hr : 0 < r) : (i.sup p).ball 0 r ∈ p.basisSets :=
  (basisSets_iff _).mpr ⟨i, _, hr, rfl⟩


theorem basisSets_singleton_mem (i : ι) {r : ℝ} (hr : 0 < r) : (p i).ball 0 r ∈ p.basisSets :=
                                        /-
                                          𝕜 : Type u_1
                                          E : Type u_5
                                          ι : Type u_8
                                          inst✝² : NormedField 𝕜
                                          inst✝¹ : AddCommGroup E
                                          inst✝ : Module 𝕜 E
                                          p : SeminormFamily 𝕜 E ι
                                          i : ι
                                          r : Real
                                          hr : LT.lt 0 r
                                          ⊢ Eq ((p i).ball 0 r) (((Singleton.singleton i).sup p).ball 0 r)
                                        -/
  (basisSets_iff _).mpr ⟨{i}, _, hr, by rw [Finset.sup_singleton]⟩
                                        /-
                                          🎉 no goals
                                        -/


theorem basisSets_nonempty [Nonempty ι] : p.basisSets.Nonempty := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    ⊢ p.basisSets.Nonempty
  -/
  let i := Classical.arbitrary ι
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    i : ι := Classical.arbitrary ι
    ⊢ p.basisSets.Nonempty
  -/
  refine nonempty_def.mpr ⟨(p i).ball 0 1, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    i : ι := Classical.arbitrary ι
    ⊢ Membership.mem p.basisSets ((p i).ball 0 1)
  -/
  exact p.basisSets_singleton_mem i zero_lt_one
  /-
    🎉 no goals
  -/


theorem basisSets_intersect (U V : Set E) (hU : U ∈ p.basisSets) (hV : V ∈ p.basisSets) :
    ∃ z ∈ p.basisSets, z ⊆ U ∩ V := by
  classical
    rcases p.basisSets_iff.mp hU with ⟨s, r₁, hr₁, hU⟩
    rcases p.basisSets_iff.mp hV with ⟨t, r₂, hr₂, hV⟩
    use ((s ∪ t).sup p).ball 0 (min r₁ r₂)
    refine ⟨p.basisSets_mem (s ∪ t) (lt_min_iff.mpr ⟨hr₁, hr₂⟩), ?_⟩
    rw [hU, hV, ball_finset_sup_eq_iInter _ _ _ (lt_min_iff.mpr ⟨hr₁, hr₂⟩),
      ball_finset_sup_eq_iInter _ _ _ hr₁, ball_finset_sup_eq_iInter _ _ _ hr₂]
    exact
      Set.subset_inter
        (Set.iInter₂_mono' fun i hi =>
          ⟨i, Finset.subset_union_left hi, ball_mono <| min_le_left _ _⟩)
        (Set.iInter₂_mono' fun i hi =>
          ⟨i, Finset.subset_union_right hi, ball_mono <| min_le_right _ _⟩)


theorem basisSets_zero (U) (hU : U ∈ p.basisSets) : (0 : E) ∈ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Membership.mem U 0
  -/
  rcases p.basisSets_iff.mp hU with ⟨ι', r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    ι' : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((ι'.sup p).ball 0 r)
    ⊢ Membership.mem U 0
  -/
  rw [hU, mem_ball_zero, map_zero]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    ι' : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((ι'.sup p).ball 0 r)
    ⊢ LT.lt 0 r
  -/
  exact hr
  /-
    🎉 no goals
  -/


theorem basisSets_add (U) (hU : U ∈ p.basisSets) :
    ∃ V ∈ p.basisSets, V + V ⊆ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Exists fun V => And (Membership.mem p.basisSets V) (HasSubset.Subset (HAdd.h …
  -/
  rcases p.basisSets_iff.mp hU with ⟨s, r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem p.basisSets V) (HasSubset.Subset (HAdd.h …
  -/
  use (s.sup p).ball 0 (r / 2)
  /-
    case h
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ And (Membership.mem p.basisSets ((s.sup p).ball 0 (HDiv.hDiv r 2))) (HasSubs …
  -/
  refine ⟨p.basisSets_mem s (div_pos hr zero_lt_two), ?_⟩
  /-
    case h
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset (HAdd.hAdd ((s.sup p).ball 0 (HDiv.hDiv r 2)) ((s.sup p).ba …
  -/
  refine Set.Subset.trans (ball_add_ball_subset (s.sup p) (r / 2) (r / 2) 0 0) ?_
  /-
    case h
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset ((s.sup p).ball (HAdd.hAdd 0 0) (HAdd.hAdd (HDiv.hDiv r 2)  …
  -/
  rw [hU, add_zero, add_halves]
  /-
    🎉 no goals
  -/


theorem basisSets_neg (U) (hU' : U ∈ p.basisSets) :
    ∃ V ∈ p.basisSets, V ⊆ (fun x : E => -x) ⁻¹' U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU' : Membership.mem p.basisSets U
    ⊢ Exists fun V => And (Membership.mem p.basisSets V) (HasSubset.Subset V (Set. …
  -/
  rcases p.basisSets_iff.mp hU' with ⟨s, r, _, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU' : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    left✝ : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem p.basisSets V) (HasSubset.Subset V (Set. …
  -/
  rw [hU, neg_preimage, neg_ball (s.sup p), neg_zero]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    U : Set E
    hU' : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    left✝ : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem p.basisSets V) (HasSubset.Subset V ((s.s …
  -/
  exact ⟨U, hU', Eq.subset hU⟩
  /-
    🎉 no goals
  -/


/-- The `addGroupFilterBasis` induced by the filter basis `Seminorm.basisSets`. -/
protected def addGroupFilterBasis [Nonempty ι] : AddGroupFilterBasis E :=
  addGroupFilterBasisOfComm p.basisSets p.basisSets_nonempty p.basisSets_intersect p.basisSets_zero
    p.basisSets_add p.basisSets_neg


theorem basisSets_smul_right (v : E) (U : Set E) (hU : U ∈ p.basisSets) :
    ∀ᶠ x : 𝕜 in 𝓝 0, x • v ∈ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x v)) (nhds 0)
  -/
  rcases p.basisSets_iff.mp hU with ⟨s, r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Filter.Eventually (fun x => Membership.mem U (HSMul.hSMul x v)) (nhds 0)
  -/
  rw [hU, Filter.eventually_iff]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Membership.mem (nhds 0) (setOf fun x => Membership.mem ((s.sup p).ball 0 r)  …
  -/
  simp_rw [(s.sup p).mem_ball_zero, map_smul_eq_mul]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Membership.mem (nhds 0) (setOf fun x => LT.lt (HMul.hMul (Norm.norm x) ((s.s …
  -/
  by_cases h : 0 < (s.sup p) v
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      v : E
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : LT.lt 0 ((s.sup p) v)
      ⊢ Membership.mem (nhds 0) (setOf fun x => LT.lt (HMul.hMul (Norm.norm x) ((s.s …
    -/
  · simp_rw [(lt_div_iff₀ h).symm]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      v : E
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : LT.lt 0 ((s.sup p) v)
      ⊢ Membership.mem (nhds 0) (setOf fun x => LT.lt (Norm.norm x) (HDiv.hDiv r ((s …
    -/
    rw [← _root_.ball_zero_eq]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝² : NormedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      v : E
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : LT.lt 0 ((s.sup p) v)
      ⊢ Membership.mem (nhds 0) (Metric.ball 0 (HDiv.hDiv r ((s.sup p) v)))
    -/
    exact Metric.ball_mem_nhds 0 (div_pos hr h)
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    h : Not (LT.lt 0 ((s.sup p) v))
    ⊢ Membership.mem (nhds 0) (setOf fun x => LT.lt (HMul.hMul (Norm.norm x) ((s.s …
  -/
  simp_rw [le_antisymm (not_lt.mp h) (apply_nonneg _ v), mul_zero, hr]
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    v : E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    h : Not (LT.lt 0 ((s.sup p) v))
    ⊢ Membership.mem (nhds 0) (setOf fun x => True)
  -/
  exact IsOpen.mem_nhds isOpen_univ (mem_univ 0)
  /-
    🎉 no goals
  -/


theorem basisSets_smul (U) (hU : U ∈ p.basisSets) :
    ∃ V ∈ 𝓝 (0 : 𝕜), ∃ W ∈ p.addGroupFilterBasis.sets, V • W ⊆ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
  -/
  rcases p.basisSets_iff.mp hU with ⟨s, r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem (nhds 0) V) (Exists fun W => And (Member …
  -/
  refine ⟨Metric.ball 0 √r, Metric.ball_mem_nhds 0 (Real.sqrt_pos.mpr hr), ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun W => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets W …
  -/
  refine ⟨(s.sup p).ball 0 √r, p.basisSets_mem s (Real.sqrt_pos.mpr hr), ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset (HSMul.hSMul (Metric.ball 0 r.sqrt) ((s.sup p).ball 0 r.sqr …
  -/
  refine Set.Subset.trans (ball_smul_ball (s.sup p) √r √r) ?_
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset ((s.sup p).ball 0 (HMul.hMul r.sqrt r.sqrt)) U
  -/
  rw [hU, Real.mul_self_sqrt (le_of_lt hr)]
  /-
    🎉 no goals
  -/


theorem basisSets_smul_left (x : 𝕜) (U : Set E) (hU : U ∈ p.basisSets) :
    ∃ V ∈ p.addGroupFilterBasis.sets, V ⊆ (fun y : E => x • y) ⁻¹' U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    x : 𝕜
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
  -/
  rcases p.basisSets_iff.mp hU with ⟨s, r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    x : 𝕜
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
  -/
  rw [hU]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    x : 𝕜
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
  -/
  by_cases h : x ≠ 0
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : Nonempty ι
      x : 𝕜
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : Ne x 0
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
  · rw [(s.sup p).smul_ball_preimage 0 r x h, smul_zero]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : Nonempty ι
      x : 𝕜
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : Ne x 0
      ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
    -/
    use (s.sup p).ball 0 (r / ‖x‖)
    /-
      case h
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : Nonempty ι
      x : 𝕜
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : Ne x 0
      ⊢ And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets ((s.sup p).ball 0 …
    -/
    exact ⟨p.basisSets_mem s (div_pos hr (norm_pos_iff.mpr h)), Subset.rfl⟩
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    x : 𝕜
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    h : Not (Ne x 0)
    ⊢ Exists fun V => And (Membership.mem AddGroupFilterBasis.toFilterBasis.sets V …
  -/
  refine ⟨(s.sup p).ball 0 r, p.basisSets_mem s hr, ?_⟩
  simp only [not_ne_iff.mp h, Set.subset_def, mem_ball_zero, hr, mem_univ, map_zero, imp_true_iff,
    preimage_const_of_mem, zero_smul]


/-- The `moduleFilterBasis` induced by the filter basis `Seminorm.basisSets`. -/
protected def moduleFilterBasis : ModuleFilterBasis 𝕜 E where
  toAddGroupFilterBasis := p.addGroupFilterBasis
  smul' := p.basisSets_smul _
  smul_left' := p.basisSets_smul_left
  smul_right' := p.basisSets_smul_right


theorem filter_eq_iInf (p : SeminormFamily 𝕜 E ι) :
    p.moduleFilterBasis.toFilterBasis.filter = ⨅ i, (𝓝 0).comap (p i) := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    ⊢ Eq AddGroupFilterBasis.toFilterBasis.filter (iInf fun i => Filter.comap (⇑(p …
  -/
  refine le_antisymm (le_iInf fun i => ?_) ?_
  · rw [p.moduleFilterBasis.toFilterBasis.hasBasis.le_basis_iff
        (Metric.nhds_basis_ball.comap _)]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      i : ι
      ⊢ ∀ (i' : Real), LT.lt 0 i' → Exists fun i_1 => And (Membership.mem AddGroupFi …
    -/
    intro ε hε
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      i : ι
      ε : Real
      hε : LT.lt 0 ε
      ⊢ Exists fun i_1 => And (Membership.mem AddGroupFilterBasis.toFilterBasis i_1) …
    -/
    refine ⟨(p i).ball 0 ε, ?_, ?_⟩
      /-
        case refine_1.refine_1
        𝕜 : Type u_1
        E : Type u_5
        ι : Type u_8
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Nonempty ι
        p : SeminormFamily 𝕜 E ι
        i : ι
        ε : Real
        hε : LT.lt 0 ε
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis ((p i).ball 0 ε)
      -/
    · rw [← (Finset.sup_singleton : _ = p i)]
      /-
        case refine_1.refine_1
        𝕜 : Type u_1
        E : Type u_5
        ι : Type u_8
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Nonempty ι
        p : SeminormFamily 𝕜 E ι
        i : ι
        ε : Real
        hε : LT.lt 0 ε
        ⊢ Membership.mem AddGroupFilterBasis.toFilterBasis (((Singleton.singleton i).s …
      -/
      exact p.basisSets_mem {i} hε
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        𝕜 : Type u_1
        E : Type u_5
        ι : Type u_8
        inst✝³ : NormedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : Nonempty ι
        p : SeminormFamily 𝕜 E ι
        i : ι
        ε : Real
        hε : LT.lt 0 ε
        ⊢ HasSubset.Subset (id ((p i).ball 0 ε)) (Set.preimage (⇑(p i)) (Metric.ball 0 …
      -/
    · rw [id, (p i).ball_zero_eq_preimage_ball]
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      ⊢ LE.le (iInf fun i => Filter.comap (⇑(p i)) (nhds 0)) AddGroupFilterBasis.toF …
    -/
  · rw [p.moduleFilterBasis.toFilterBasis.hasBasis.ge_iff]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      ⊢ ∀ (i' : Set E), Membership.mem AddGroupFilterBasis.toFilterBasis i' → Member …
    -/
    rintro U (hU : U ∈ p.basisSets)
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      U : Set E
      hU : Membership.mem p.basisSets U
      ⊢ Membership.mem (iInf fun i => Filter.comap (⇑(p i)) (nhds 0)) (id U)
    -/
    rcases p.basisSets_iff.mp hU with ⟨s, r, hr, rfl⟩
    /-
      case refine_2.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hU : Membership.mem p.basisSets ((s.sup p).ball 0 r)
      ⊢ Membership.mem (iInf fun i => Filter.comap (⇑(p i)) (nhds 0)) (id ((s.sup p) …
    -/
    rw [id, Seminorm.ball_finset_sup_eq_iInter _ _ _ hr, s.iInter_mem_sets]
    exact fun i _ =>
      Filter.mem_iInf_of_mem i
        ⟨Metric.ball 0 r, Metric.ball_mem_nhds 0 hr,
          Eq.subset (p i).ball_zero_eq_preimage_ball.symm⟩


/-- If a family of seminorms is continuous, then their basis sets are neighborhoods of zero. -/
lemma basisSets_mem_nhds {𝕜 E ι : Type*} [NormedField 𝕜]
    [AddCommGroup E] [Module 𝕜 E] [TopologicalSpace E] (p : SeminormFamily 𝕜 E ι)
    (hp : ∀ i, Continuous (p i)) (U : Set E) (hU : U ∈ p.basisSets) : U ∈ 𝓝 (0 : E) := by
  /-
    𝕜 : Type u_10
    E : Type u_11
    ι : Type u_12
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : ∀ (i : ι), Continuous ⇑(p i)
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Membership.mem (nhds 0) U
  -/
  obtain ⟨s, r, hr, rfl⟩ := p.basisSets_iff.mp hU
  /-
    case intro.intro.intro
    𝕜 : Type u_10
    E : Type u_11
    ι : Type u_12
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : ∀ (i : ι), Continuous ⇑(p i)
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    hU : Membership.mem p.basisSets ((s.sup p).ball 0 r)
    ⊢ Membership.mem (nhds 0) ((s.sup p).ball 0 r)
  -/
  clear hU
  /-
    case intro.intro.intro
    𝕜 : Type u_10
    E : Type u_11
    ι : Type u_12
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : ∀ (i : ι), Continuous ⇑(p i)
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    ⊢ Membership.mem (nhds 0) ((s.sup p).ball 0 r)
  -/
  refine Seminorm.ball_mem_nhds ?_ hr
  classical
  induction s using Finset.induction_on
  case empty => simpa using continuous_zero
  case insert a s _ hs =>
    simp only [Finset.sup_insert, coe_sup]
    exact Continuous.max (hp a) hs


/-- The proposition that a linear map is bounded between spaces with families of seminorms. -/
def IsBounded (p : ι → Seminorm 𝕜 E) (q : ι' → Seminorm 𝕜₂ F) (f : E →ₛₗ[σ₁₂] F) : Prop :=
  ∀ i, ∃ s : Finset ι, ∃ C : ℝ≥0, (q i).comp f ≤ C • s.sup p


theorem isBounded_const (ι' : Type*) [Nonempty ι'] {p : ι → Seminorm 𝕜 E} {q : Seminorm 𝕜₂ F}
    (f : E →ₛₗ[σ₁₂] F) :
    IsBounded p (fun _ : ι' => q) f ↔ ∃ (s : Finset ι) (C : ℝ≥0), q.comp f ≤ C • s.sup p := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι' : Type u_10
    inst✝ : Nonempty ι'
    p : ι → Seminorm 𝕜 E
    q : Seminorm 𝕜₂ F
    f : LinearMap σ₁₂ E F
    ⊢ Iff (Seminorm.IsBounded p (fun x => q) f) (Exists fun s => Exists fun C => L …
  -/
  simp only [IsBounded, forall_const]
  /-
    🎉 no goals
  -/


theorem const_isBounded (ι : Type*) [Nonempty ι] {p : Seminorm 𝕜 E} {q : ι' → Seminorm 𝕜₂ F}
    (f : E →ₛₗ[σ₁₂] F) : IsBounded (fun _ : ι => p) q f ↔ ∀ i, ∃ C : ℝ≥0, (q i).comp f ≤ C • p := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_10
    inst✝ : Nonempty ι
    p : Seminorm 𝕜 E
    q : ι' → Seminorm 𝕜₂ F
    f : LinearMap σ₁₂ E F
    ⊢ Iff (Seminorm.IsBounded (fun x => p) q f) (∀ (i : ι'), Exists fun C => LE.le …
  -/
  constructor <;> intro h i
    /-
      case mp
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_5
      F : Type u_6
      ι' : Type u_9
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : NormedField 𝕜₂
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝¹ : RingHomIsometric σ₁₂
      ι : Type u_10
      inst✝ : Nonempty ι
      p : Seminorm 𝕜 E
      q : ι' → Seminorm 𝕜₂ F
      f : LinearMap σ₁₂ E F
      h : Seminorm.IsBounded (fun x => p) q f
      i : ι'
      ⊢ Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C p)
    -/
  · rcases h i with ⟨s, C, h⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      𝕜₂ : Type u_2
      E : Type u_5
      F : Type u_6
      ι' : Type u_9
      inst✝⁷ : NormedField 𝕜
      inst✝⁶ : AddCommGroup E
      inst✝⁵ : Module 𝕜 E
      inst✝⁴ : NormedField 𝕜₂
      inst✝³ : AddCommGroup F
      inst✝² : Module 𝕜₂ F
      σ₁₂ : RingHom 𝕜 𝕜₂
      inst✝¹ : RingHomIsometric σ₁₂
      ι : Type u_10
      inst✝ : Nonempty ι
      p : Seminorm 𝕜 E
      q : ι' → Seminorm 𝕜₂ F
      f : LinearMap σ₁₂ E F
      h✝ : Seminorm.IsBounded (fun x => p) q f
      i : ι'
      s : Finset ι
      C : NNReal
      h : LE.le ((q i).comp f) (HSMul.hSMul C (s.sup fun x => p))
      ⊢ Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C p)
    -/
    exact ⟨C, le_trans h (smul_le_smul (Finset.sup_le fun _ _ => le_rfl) le_rfl)⟩
    /-
      🎉 no goals
    -/
  /-
    case mpr
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_10
    inst✝ : Nonempty ι
    p : Seminorm 𝕜 E
    q : ι' → Seminorm 𝕜₂ F
    f : LinearMap σ₁₂ E F
    h : ∀ (i : ι'), Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C p)
    i : ι'
    ⊢ Exists fun s => Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C (s.sup f …
  -/
  use {Classical.arbitrary ι}
  /-
    case h
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    ι : Type u_10
    inst✝ : Nonempty ι
    p : Seminorm 𝕜 E
    q : ι' → Seminorm 𝕜₂ F
    f : LinearMap σ₁₂ E F
    h : ∀ (i : ι'), Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C p)
    i : ι'
    ⊢ Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C ((Singleton.singleton (C …
  -/
  simp only [h, Finset.sup_singleton]
  /-
    🎉 no goals
  -/


theorem isBounded_sup {p : ι → Seminorm 𝕜 E} {q : ι' → Seminorm 𝕜₂ F} {f : E →ₛₗ[σ₁₂] F}
    (hf : IsBounded p q f) (s' : Finset ι') :
    ∃ (C : ℝ≥0) (s : Finset ι), (s'.sup q).comp f ≤ C • s.sup p := by
  classical
    obtain rfl | _ := s'.eq_empty_or_nonempty
    · exact ⟨1, ∅, by simp [Seminorm.bot_eq_zero]⟩
    choose fₛ fC hf using hf
    use s'.card • s'.sup fC, Finset.biUnion s' fₛ
    have hs : ∀ i : ι', i ∈ s' → (q i).comp f ≤ s'.sup fC • (Finset.biUnion s' fₛ).sup p := by
      intro i hi
      refine (hf i).trans (smul_le_smul ?_ (Finset.le_sup hi))
      exact Finset.sup_mono (Finset.subset_biUnion_of_mem fₛ hi)
    refine (comp_mono f (finset_sup_le_sum q s')).trans ?_
    simp_rw [← pullback_apply, map_sum, pullback_apply]
    refine (Finset.sum_le_sum hs).trans ?_
    rw [Finset.sum_const, smul_assoc]


/-- The proposition that the topology of `E` is induced by a family of seminorms `p`. -/
structure WithSeminorms (p : SeminormFamily 𝕜 E ι) [topology : TopologicalSpace E] : Prop where
  topology_eq_withSeminorms : topology = p.moduleFilterBasis.topology


theorem WithSeminorms.withSeminorms_eq {p : SeminormFamily 𝕜 E ι} [t : TopologicalSpace E]
    (hp : WithSeminorms p) : t = p.moduleFilterBasis.topology :=
  hp.1


theorem WithSeminorms.topologicalAddGroup (hp : WithSeminorms p) : TopologicalAddGroup E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ TopologicalAddGroup E
  -/
  rw [hp.withSeminorms_eq]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ TopologicalAddGroup E
  -/
  exact AddGroupFilterBasis.isTopologicalAddGroup _
  /-
    🎉 no goals
  -/


theorem WithSeminorms.continuousSMul (hp : WithSeminorms p) : ContinuousSMul 𝕜 E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ ContinuousSMul 𝕜 E
  -/
  rw [hp.withSeminorms_eq]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ ContinuousSMul 𝕜 E
  -/
  exact ModuleFilterBasis.continuousSMul _
  /-
    🎉 no goals
  -/


theorem WithSeminorms.hasBasis (hp : WithSeminorms p) :
    (𝓝 (0 : E)).HasBasis (fun s : Set E => s ∈ p.basisSets) id := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ (nhds 0).HasBasis (fun s => Membership.mem p.basisSets s) id
  -/
  rw [congr_fun (congr_arg (@nhds E) hp.1) 0]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ (nhds 0).HasBasis (fun s => Membership.mem p.basisSets s) id
  -/
  exact AddGroupFilterBasis.nhds_zero_hasBasis _
  /-
    🎉 no goals
  -/


theorem WithSeminorms.hasBasis_zero_ball (hp : WithSeminorms p) :
    (𝓝 (0 : E)).HasBasis
    (fun sr : Finset ι × ℝ => 0 < sr.2) fun sr => (sr.1.sup p).ball 0 sr.2 := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ (nhds 0).HasBasis (fun sr => LT.lt 0 sr.2) fun sr => (sr.1.sup p).ball 0 sr.2
  -/
  refine ⟨fun V => ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    V : Set E
    ⊢ Iff (Membership.mem (nhds 0) V) (Exists fun i => And (LT.lt 0 i.2) (HasSubse …
  -/
  simp only [hp.hasBasis.mem_iff, SeminormFamily.basisSets_iff, Prod.exists]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    V : Set E
    ⊢ Iff (Exists fun i => And (Exists fun i_1 => Exists fun r => And (LT.lt 0 r)  …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      V : Set E
      ⊢ (Exists fun i => And (Exists fun i_1 => Exists fun r => And (LT.lt 0 r) (Eq  …
    -/
  · rintro ⟨-, ⟨s, r, hr, rfl⟩, hV⟩
    /-
      case mp.intro.intro.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      V : Set E
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hV : HasSubset.Subset (id ((s.sup p).ball 0 r)) V
      ⊢ Exists fun a => Exists fun b => And (LT.lt 0 b) (HasSubset.Subset ((a.sup p) …
    -/
    exact ⟨s, r, hr, hV⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      V : Set E
      ⊢ (Exists fun a => Exists fun b => And (LT.lt 0 b) (HasSubset.Subset ((a.sup p …
    -/
  · rintro ⟨s, r, hr, hV⟩
    /-
      case mpr.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      V : Set E
      s : Finset ι
      r : Real
      hr : LT.lt 0 r
      hV : HasSubset.Subset ((s.sup p).ball 0 r) V
      ⊢ Exists fun i => And (Exists fun i_1 => Exists fun r => And (LT.lt 0 r) (Eq i …
    -/
    exact ⟨_, ⟨s, r, hr, rfl⟩, hV⟩
    /-
      🎉 no goals
    -/


theorem WithSeminorms.hasBasis_ball (hp : WithSeminorms p) {x : E} :
    (𝓝 (x : E)).HasBasis
    (fun sr : Finset ι × ℝ => 0 < sr.2) fun sr => (sr.1.sup p).ball x sr.2 := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    ⊢ (nhds x).HasBasis (fun sr => LT.lt 0 sr.2) fun sr => (sr.1.sup p).ball x sr.2
  -/
  have : TopologicalAddGroup E := hp.topologicalAddGroup
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    this : TopologicalAddGroup E
    ⊢ (nhds x).HasBasis (fun sr => LT.lt 0 sr.2) fun sr => (sr.1.sup p).ball x sr.2
  -/
  rw [← map_add_left_nhds_zero]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    this : TopologicalAddGroup E
    ⊢ (Filter.map (fun x_1 => HAdd.hAdd x x_1) (nhds 0)).HasBasis (fun sr => LT.lt …
  -/
  convert hp.hasBasis_zero_ball.map (x + ·) using 1
  /-
    case h.e'_5
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    this : TopologicalAddGroup E
    ⊢ Eq (fun sr => (sr.1.sup p).ball x sr.2) fun i => Set.image (fun x_1 => HAdd. …
  -/
  ext sr : 1
  -- Porting note: extra type ascriptions needed on `0`
  have : (sr.fst.sup p).ball (x +ᵥ (0 : E)) sr.snd = x +ᵥ (sr.fst.sup p).ball 0 sr.snd :=
    Eq.symm (Seminorm.vadd_ball (sr.fst.sup p))
  /-
    case h.e'_5.h
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    this✝ : TopologicalAddGroup E
    sr : Prod (Finset ι) Real
    this : Eq ((sr.1.sup p).ball (HVAdd.hVAdd x 0) sr.2) (HVAdd.hVAdd x ((sr.1.sup …
    ⊢ Eq ((sr.1.sup p).ball x sr.2) (Set.image (fun x_1 => HAdd.hAdd x x_1) ((sr.1 …
  -/
  rwa [vadd_eq_add, add_zero] at this
  /-
    🎉 no goals
  -/


/-- The `x`-neighbourhoods of a space whose topology is induced by a family of seminorms
are exactly the sets which contain seminorm balls around `x`. -/
theorem WithSeminorms.mem_nhds_iff (hp : WithSeminorms p) (x : E) (U : Set E) :
    U ∈ 𝓝 x ↔ ∃ s : Finset ι, ∃ r > 0, (s.sup p).ball x r ⊆ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    x : E
    U : Set E
    ⊢ Iff (Membership.mem (nhds x) U) (Exists fun s => Exists fun r => And (GT.gt  …
  -/
  rw [hp.hasBasis_ball.mem_iff, Prod.exists]
  /-
    🎉 no goals
  -/


/-- The open sets of a space whose topology is induced by a family of seminorms
are exactly the sets which contain seminorm balls around all of their points. -/
theorem WithSeminorms.isOpen_iff_mem_balls (hp : WithSeminorms p) (U : Set E) :
    IsOpen U ↔ ∀ x ∈ U, ∃ s : Finset ι, ∃ r > 0, (s.sup p).ball x r ⊆ U := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    U : Set E
    ⊢ Iff (IsOpen U) (∀ (x : E), Membership.mem U x → Exists fun s => Exists fun r …
  -/
  simp_rw [← WithSeminorms.mem_nhds_iff hp _ U, isOpen_iff_mem_nhds]
  /-
    🎉 no goals
  -/

/- Note that through the following lemmas, one also immediately has that separating families
of seminorms induce T₂ and T₃ topologies by `TopologicalAddGroup.t2Space`
and `TopologicalAddGroup.t3Space` -/

/-- A separating family of seminorms induces a T₁ topology. -/
theorem WithSeminorms.T1_of_separating (hp : WithSeminorms p)
    (h : ∀ x, x ≠ 0 → ∃ i, p i x ≠ 0) : T1Space E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    ⊢ T1Space E
  -/
  have := hp.topologicalAddGroup
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    ⊢ T1Space E
  -/
  refine TopologicalAddGroup.t1Space _ ?_
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    ⊢ IsClosed (Singleton.singleton 0)
  -/
  rw [← isOpen_compl_iff, hp.isOpen_iff_mem_balls]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    ⊢ ∀ (x : E), Membership.mem (HasCompl.compl (Singleton.singleton 0)) x → Exist …
  -/
  rintro x (hx : x ≠ 0)
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    x : E
    hx : Ne x 0
    ⊢ Exists fun s => Exists fun r => And (GT.gt r 0) (HasSubset.Subset ((s.sup p) …
  -/
  cases' h x hx with i pi_nonzero
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    x : E
    hx : Ne x 0
    i : ι
    pi_nonzero : Ne ((p i) x) 0
    ⊢ Exists fun s => Exists fun r => And (GT.gt r 0) (HasSubset.Subset ((s.sup p) …
  -/
  refine ⟨{i}, p i x, by positivity, subset_compl_singleton_iff.mpr ?_⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    h : ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
    this : TopologicalAddGroup E
    x : E
    hx : Ne x 0
    i : ι
    pi_nonzero : Ne ((p i) x) 0
    ⊢ Not (Membership.mem (((Singleton.singleton i).sup p).ball x ((p i) x)) 0)
  -/
  rw [Finset.sup_singleton, mem_ball, zero_sub, map_neg_eq_map, not_lt]
  /-
    🎉 no goals
  -/


/-- A family of seminorms inducing a T₁ topology is separating. -/
theorem WithSeminorms.separating_of_T1 [T1Space E] (hp : WithSeminorms p) (x : E) (hx : x ≠ 0) :
    ∃ i, p i x ≠ 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    ⊢ Exists fun i => Ne ((p i) x) 0
  -/
  have := ((t1Space_TFAE E).out 0 9).mp (inferInstanceAs <| T1Space E)
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    this : ∀ ⦃x y : E⦄, Specializes x y → Eq x y
    ⊢ Exists fun i => Ne ((p i) x) 0
  -/
  by_contra! h
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    this : ∀ ⦃x y : E⦄, Specializes x y → Eq x y
    h : ∀ (i : ι), Eq ((p i) x) 0
    ⊢ False
  -/
  refine hx (this ?_)
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    this : ∀ ⦃x y : E⦄, Specializes x y → Eq x y
    h : ∀ (i : ι), Eq ((p i) x) 0
    ⊢ Specializes x 0
  -/
  rw [hp.hasBasis_zero_ball.specializes_iff]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    this : ∀ ⦃x y : E⦄, Specializes x y → Eq x y
    h : ∀ (i : ι), Eq ((p i) x) 0
    ⊢ ∀ (i : Prod (Finset ι) Real), LT.lt 0 i.2 → Membership.mem ((i.1.sup p).ball …
  -/
  rintro ⟨s, r⟩ (hr : 0 < r)
  /-
    case mk
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝ : T1Space E
    hp : WithSeminorms p
    x : E
    hx : Ne x 0
    this : ∀ ⦃x y : E⦄, Specializes x y → Eq x y
    h : ∀ (i : ι), Eq ((p i) x) 0
    s : Finset ι
    r : Real
    hr : LT.lt 0 r
    ⊢ Membership.mem (({ fst := s, snd := r }.1.sup p).ball 0 { fst := s, snd := r …
  -/
  simp only [ball_finset_sup_eq_iInter _ _ _ hr, mem_iInter₂, mem_ball_zero, h, hr, forall_true_iff]
  /-
    🎉 no goals
  -/


/-- A family of seminorms is separating iff it induces a T₁ topology. -/
theorem WithSeminorms.separating_iff_T1 (hp : WithSeminorms p) :
    (∀ x, x ≠ 0 → ∃ i, p i x ≠ 0) ↔ T1Space E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ Iff (∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0) (T1Space E)
  -/
  refine ⟨WithSeminorms.T1_of_separating hp, ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ T1Space E → ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
  -/
  intro
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    a✝ : T1Space E
    ⊢ ∀ (x : E), Ne x 0 → Exists fun i => Ne ((p i) x) 0
  -/
  exact WithSeminorms.separating_of_T1 hp
  /-
    🎉 no goals
  -/


/-- Convergence along filters for `WithSeminorms`.

Variant with `Finset.sup`. -/
theorem WithSeminorms.tendsto_nhds' (hp : WithSeminorms p) (u : F → E) {f : Filter F} (y₀ : E) :
    Filter.Tendsto u f (𝓝 y₀) ↔
    ∀ (s : Finset ι) (ε), 0 < ε → ∀ᶠ x in f, s.sup p (u x - y₀) < ε := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    u : F → E
    f : Filter F
    y₀ : E
    ⊢ Iff (Filter.Tendsto u f (nhds y₀)) (∀ (s : Finset ι) (ε : Real), LT.lt 0 ε → …
  -/
  simp [hp.hasBasis_ball.tendsto_right_iff]
  /-
    🎉 no goals
  -/


/-- Convergence along filters for `WithSeminorms`. -/
theorem WithSeminorms.tendsto_nhds (hp : WithSeminorms p) (u : F → E) {f : Filter F} (y₀ : E) :
    Filter.Tendsto u f (𝓝 y₀) ↔ ∀ i ε, 0 < ε → ∀ᶠ x in f, p i (u x - y₀) < ε := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    u : F → E
    f : Filter F
    y₀ : E
    ⊢ Iff (Filter.Tendsto u f (nhds y₀)) (∀ (i : ι) (ε : Real), LT.lt 0 ε → Filter …
  -/
  rw [hp.tendsto_nhds' u y₀]
  exact
    ⟨fun h i => by simpa only [Finset.sup_singleton] using h {i}, fun h s ε hε =>
      (s.eventually_all.2 fun i _ => h i ε hε).mono fun _ => finset_sup_apply_lt hε⟩


/-- Limit `→ ∞` for `WithSeminorms`. -/
theorem WithSeminorms.tendsto_nhds_atTop (hp : WithSeminorms p) (u : F → E) (y₀ : E) :
    Filter.Tendsto u Filter.atTop (𝓝 y₀) ↔
    ∀ i ε, 0 < ε → ∃ x₀, ∀ x, x₀ ≤ x → p i (u x - y₀) < ε := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nonempty ι
    inst✝² : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝¹ : SemilatticeSup F
    inst✝ : Nonempty F
    hp : WithSeminorms p
    u : F → E
    y₀ : E
    ⊢ Iff (Filter.Tendsto u Filter.atTop (nhds y₀)) (∀ (i : ι) (ε : Real), LT.lt 0 …
  -/
  rw [hp.tendsto_nhds u y₀]
  /-
    𝕜 : Type u_1
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : Nonempty ι
    inst✝² : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    inst✝¹ : SemilatticeSup F
    inst✝ : Nonempty F
    hp : WithSeminorms p
    u : F → E
    y₀ : E
    ⊢ Iff (∀ (i : ι) (ε : Real), LT.lt 0 ε → Filter.Eventually (fun x => LT.lt ((p …
  -/
  exact forall₃_congr fun _ _ _ => Filter.eventually_atTop
  /-
    🎉 no goals
  -/


theorem SeminormFamily.withSeminorms_of_nhds [TopologicalAddGroup E] (p : SeminormFamily 𝕜 E ι)
    (h : 𝓝 (0 : E) = p.moduleFilterBasis.toFilterBasis.filter) : WithSeminorms p := by
  refine
    ⟨TopologicalAddGroup.ext inferInstance p.addGroupFilterBasis.isTopologicalAddGroup ?_⟩
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    h : Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter
    ⊢ Eq (nhds 0) (nhds 0)
  -/
  rw [AddGroupFilterBasis.nhds_zero_eq]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    h : Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter
    ⊢ Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem SeminormFamily.withSeminorms_of_hasBasis [TopologicalAddGroup E] (p : SeminormFamily 𝕜 E ι)
    (h : (𝓝 (0 : E)).HasBasis (fun s : Set E => s ∈ p.basisSets) id) : WithSeminorms p :=
  p.withSeminorms_of_nhds <|
    Filter.HasBasis.eq_of_same_basis h p.addGroupFilterBasis.toFilterBasis.hasBasis


theorem SeminormFamily.withSeminorms_iff_nhds_eq_iInf [TopologicalAddGroup E]
    (p : SeminormFamily 𝕜 E ι) : WithSeminorms p ↔ (𝓝 (0 : E)) = ⨅ i, (𝓝 0).comap (p i) := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    ⊢ Iff (WithSeminorms p) (Eq (nhds 0) (iInf fun i => Filter.comap (⇑(p i)) (nhd …
  -/
  rw [← p.filter_eq_iInf]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    ⊢ Iff (WithSeminorms p) (Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter)
  -/
  refine ⟨fun h => ?_, p.withSeminorms_of_nhds⟩
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    h : WithSeminorms p
    ⊢ Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter
  -/
  rw [h.topology_eq_withSeminorms]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    h : WithSeminorms p
    ⊢ Eq (nhds 0) AddGroupFilterBasis.toFilterBasis.filter
  -/
  exact AddGroupFilterBasis.nhds_zero_eq _
  /-
    🎉 no goals
  -/


/-- The topology induced by a family of seminorms is exactly the infimum of the ones induced by
each seminorm individually. We express this as a characterization of `WithSeminorms p`. -/
theorem SeminormFamily.withSeminorms_iff_topologicalSpace_eq_iInf [TopologicalAddGroup E]
    (p : SeminormFamily 𝕜 E ι) :
    WithSeminorms p ↔
      t = ⨅ i, (p i).toSeminormedAddCommGroup.toUniformSpace.toTopologicalSpace := by
  rw [p.withSeminorms_iff_nhds_eq_iInf,
    TopologicalAddGroup.ext_iff inferInstance (topologicalAddGroup_iInf fun i => inferInstance),
    nhds_iInf]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    ⊢ Iff (Eq (nhds 0) (iInf fun i => Filter.comap (⇑(p i)) (nhds 0))) (Eq (nhds 0 …
  -/
  congrm _ = ⨅ i, ?_
  /-
    case a
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    t : TopologicalSpace E
    inst✝ : TopologicalAddGroup E
    p : SeminormFamily 𝕜 E ι
    i : ι
    ⊢ Eq (Filter.comap (⇑(p i)) (nhds 0)) (nhds 0)
  -/
  exact @comap_norm_nhds_zero _ (p i).toSeminormedAddGroup
  /-
    🎉 no goals
  -/


theorem WithSeminorms.continuous_seminorm {p : SeminormFamily 𝕜 E ι} (hp : WithSeminorms p)
    (i : ι) : Continuous (p i) := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    i : ι
    ⊢ Continuous ⇑(p i)
  -/
  have := hp.topologicalAddGroup
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    i : ι
    this : TopologicalAddGroup E
    ⊢ Continuous ⇑(p i)
  -/
  rw [p.withSeminorms_iff_topologicalSpace_eq_iInf.mp hp]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    i : ι
    this : TopologicalAddGroup E
    ⊢ Continuous ⇑(p i)
  -/
  exact continuous_iInf_dom (@continuous_norm _ (p i).toSeminormedAddGroup)
  /-
    🎉 no goals
  -/


/-- The uniform structure induced by a family of seminorms is exactly the infimum of the ones
induced by each seminorm individually. We express this as a characterization of
`WithSeminorms p`. -/
theorem SeminormFamily.withSeminorms_iff_uniformSpace_eq_iInf [u : UniformSpace E]
    [UniformAddGroup E] (p : SeminormFamily 𝕜 E ι) :
    WithSeminorms p ↔ u = ⨅ i, (p i).toSeminormedAddCommGroup.toUniformSpace := by
  rw [p.withSeminorms_iff_nhds_eq_iInf,
    UniformAddGroup.ext_iff inferInstance (uniformAddGroup_iInf fun i => inferInstance),
    UniformSpace.toTopologicalSpace_iInf, nhds_iInf]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    u : UniformSpace E
    inst✝ : UniformAddGroup E
    p : SeminormFamily 𝕜 E ι
    ⊢ Iff (Eq (nhds 0) (iInf fun i => Filter.comap (⇑(p i)) (nhds 0))) (Eq (nhds 0 …
  -/
  congrm _ = ⨅ i, ?_
  /-
    case a
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    u : UniformSpace E
    inst✝ : UniformAddGroup E
    p : SeminormFamily 𝕜 E ι
    i : ι
    ⊢ Eq (Filter.comap (⇑(p i)) (nhds 0)) (nhds 0)
  -/
  exact @comap_norm_nhds_zero _ (p i).toAddGroupSeminorm.toSeminormedAddGroup
  /-
    🎉 no goals
  -/


/-- The topology of a `NormedSpace 𝕜 E` is induced by the seminorm `normSeminorm 𝕜 E`. -/
theorem norm_withSeminorms (𝕜 E) [NormedField 𝕜] [SeminormedAddCommGroup E] [NormedSpace 𝕜 E] :
    WithSeminorms fun _ : Fin 1 => normSeminorm 𝕜 E := by
  /-
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    ⊢ WithSeminorms fun x => normSeminorm 𝕜 E
  -/
  let p : SeminormFamily 𝕜 E (Fin 1) := fun _ => normSeminorm 𝕜 E
  refine
    ⟨SeminormedAddCommGroup.toTopologicalAddGroup.ext
        p.addGroupFilterBasis.isTopologicalAddGroup ?_⟩
  /-
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    ⊢ Eq (nhds 0) (nhds 0)
  -/
  refine Filter.HasBasis.eq_of_same_basis Metric.nhds_basis_ball ?_
  /-
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    ⊢ (nhds 0).HasBasis (fun x => LT.lt 0 x) (Metric.ball 0)
  -/
  rw [← ball_normSeminorm 𝕜 E]
  refine
    Filter.HasBasis.to_hasBasis p.addGroupFilterBasis.nhds_zero_hasBasis ?_ fun r hr =>
      ⟨(normSeminorm 𝕜 E).ball 0 r, p.basisSets_singleton_mem 0 hr, rfl.subset⟩
  /-
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    ⊢ ∀ (i : Set E), Membership.mem p.addGroupFilterBasis i → Exists fun i' => And …
  -/
  rintro U (hU : U ∈ p.basisSets)
  /-
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU : Membership.mem p.basisSets U
    ⊢ Exists fun i' => And (LT.lt 0 i') (HasSubset.Subset ((normSeminorm 𝕜 E).ball …
  -/
  rcases p.basisSets_iff.mp hU with ⟨s, r, hr, hU⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset (Fin 1)
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ Exists fun i' => And (LT.lt 0 i') (HasSubset.Subset ((normSeminorm 𝕜 E).ball …
  -/
  use r, hr
  /-
    case right
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset (Fin 1)
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset ((normSeminorm 𝕜 E).ball 0 r) (id U)
  -/
  rw [hU, id]
  /-
    case right
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset (Fin 1)
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    ⊢ HasSubset.Subset ((normSeminorm 𝕜 E).ball 0 r) ((s.sup p).ball 0 r)
  -/
  by_cases h : s.Nonempty
    /-
      case pos
      𝕜 : Type u_10
      E : Type u_11
      inst✝² : NormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup E
      inst✝ : NormedSpace 𝕜 E
      p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
      U : Set E
      hU✝ : Membership.mem p.basisSets U
      s : Finset (Fin 1)
      r : Real
      hr : LT.lt 0 r
      hU : Eq U ((s.sup p).ball 0 r)
      h : s.Nonempty
      ⊢ HasSubset.Subset ((normSeminorm 𝕜 E).ball 0 r) ((s.sup p).ball 0 r)
    -/
  · rw [Finset.sup_const h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset (Fin 1)
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    h : Not s.Nonempty
    ⊢ HasSubset.Subset ((normSeminorm 𝕜 E).ball 0 r) ((s.sup p).ball 0 r)
  -/
  rw [Finset.not_nonempty_iff_eq_empty.mp h, Finset.sup_empty, ball_bot _ hr]
  /-
    case neg
    𝕜 : Type u_10
    E : Type u_11
    inst✝² : NormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup E
    inst✝ : NormedSpace 𝕜 E
    p : SeminormFamily 𝕜 E (Fin 1) := fun x => normSeminorm 𝕜 E
    U : Set E
    hU✝ : Membership.mem p.basisSets U
    s : Finset (Fin 1)
    r : Real
    hr : LT.lt 0 r
    hU : Eq U ((s.sup p).ball 0 r)
    h : Not s.Nonempty
    ⊢ HasSubset.Subset ((normSeminorm 𝕜 E).ball 0 r) Set.univ
  -/
  exact Set.subset_univ _
  /-
    🎉 no goals
  -/


theorem WithSeminorms.isVonNBounded_iff_finset_seminorm_bounded {s : Set E} (hp : WithSeminorms p) :
    Bornology.IsVonNBounded 𝕜 s ↔ ∀ I : Finset ι, ∃ r > 0, ∀ x ∈ s, I.sup p x < r := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (∀ (I : Finset ι), Exists fun r => And (GT …
  -/
  rw [hp.hasBasis.isVonNBounded_iff]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ Iff (∀ (i : Set E), Membership.mem p.basisSets i → Absorbs 𝕜 (id i) s) (∀ (I …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      ⊢ (∀ (i : Set E), Membership.mem p.basisSets i → Absorbs 𝕜 (id i) s) → ∀ (I :  …
    -/
  · intro h I
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      h : ∀ (i : Set E), Membership.mem p.basisSets i → Absorbs 𝕜 (id i) s
      I : Finset ι
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    simp only [id] at h
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      h : ∀ (i : Set E), Membership.mem p.basisSets i → Absorbs 𝕜 i s
      I : Finset ι
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    specialize h ((I.sup p).ball 0 1) (p.basisSets_mem I zero_lt_one)
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    rcases h.exists_pos with ⟨r, hr, h⟩
    /-
      case mp.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      h : ∀ (c : 𝕜), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c ((I.s …
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    cases' NormedField.exists_lt_norm 𝕜 r with a ha
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      h : ∀ (c : 𝕜), LE.le r (Norm.norm c) → HasSubset.Subset s (HSMul.hSMul c ((I.s …
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    specialize h a (le_of_lt ha)
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      h : HasSubset.Subset s (HSMul.hSMul a ((I.sup p).ball 0 1))
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    rw [Seminorm.smul_ball_zero (norm_pos_iff.1 <| hr.trans ha), mul_one] at h
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      h : HasSubset.Subset s ((I.sup p).ball 0 (Norm.norm a))
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    refine ⟨‖a‖, lt_trans hr ha, ?_⟩
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      h : HasSubset.Subset s ((I.sup p).ball 0 (Norm.norm a))
      ⊢ ∀ (x : E), Membership.mem s x → LT.lt ((I.sup p) x) (Norm.norm a)
    -/
    intro x hx
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      h : HasSubset.Subset s ((I.sup p).ball 0 (Norm.norm a))
      x : E
      hx : Membership.mem s x
      ⊢ LT.lt ((I.sup p) x) (Norm.norm a)
    -/
    specialize h hx
    /-
      case mp.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      h✝ : Absorbs 𝕜 ((I.sup p).ball 0 1) s
      r : Real
      hr : GT.gt r 0
      a : 𝕜
      ha : LT.lt r (Norm.norm a)
      x : E
      hx : Membership.mem s x
      h : Membership.mem ((I.sup p).ball 0 (Norm.norm a)) x
      ⊢ LT.lt ((I.sup p) x) (Norm.norm a)
    -/
    exact (Finset.sup I p).mem_ball_zero.mp h
    /-
      🎉 no goals
    -/
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ (∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.me …
  -/
  intro h s' hs'
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs' : Membership.mem p.basisSets s'
    ⊢ Absorbs 𝕜 (id s') s
  -/
  rcases p.basisSets_iff.mp hs' with ⟨I, r, hr, hs'⟩
  /-
    case mpr.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs'✝ : Membership.mem p.basisSets s'
    I : Finset ι
    r : Real
    hr : LT.lt 0 r
    hs' : Eq s' ((I.sup p).ball 0 r)
    ⊢ Absorbs 𝕜 (id s') s
  -/
  rw [id, hs']
  /-
    case mpr.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs'✝ : Membership.mem p.basisSets s'
    I : Finset ι
    r : Real
    hr : LT.lt 0 r
    hs' : Eq s' ((I.sup p).ball 0 r)
    ⊢ Absorbs 𝕜 ((I.sup p).ball 0 r) s
  -/
  rcases h I with ⟨r', _, h'⟩
  /-
    case mpr.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs'✝ : Membership.mem p.basisSets s'
    I : Finset ι
    r : Real
    hr : LT.lt 0 r
    hs' : Eq s' ((I.sup p).ball 0 r)
    r' : Real
    left✝ : GT.gt r' 0
    h' : ∀ (x : E), Membership.mem s x → LT.lt ((I.sup p) x) r'
    ⊢ Absorbs 𝕜 ((I.sup p).ball 0 r) s
  -/
  simp_rw [← (I.sup p).mem_ball_zero] at h'
  /-
    case mpr.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs'✝ : Membership.mem p.basisSets s'
    I : Finset ι
    r : Real
    hr : LT.lt 0 r
    hs' : Eq s' ((I.sup p).ball 0 r)
    r' : Real
    left✝ : GT.gt r' 0
    h' : ∀ (x : E), Membership.mem s x → Membership.mem ((I.sup p).ball 0 r') x
    ⊢ Absorbs 𝕜 ((I.sup p).ball 0 r) s
  -/
  refine Absorbs.mono_right ?_ h'
  /-
    case mpr.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    h : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.m …
    s' : Set E
    hs'✝ : Membership.mem p.basisSets s'
    I : Finset ι
    r : Real
    hr : LT.lt 0 r
    hs' : Eq s' ((I.sup p).ball 0 r)
    r' : Real
    left✝ : GT.gt r' 0
    h' : ∀ (x : E), Membership.mem s x → Membership.mem ((I.sup p).ball 0 r') x
    ⊢ Absorbs 𝕜 ((I.sup p).ball 0 r) ((I.sup p).ball 0 r')
  -/
  exact (Finset.sup I p).ball_zero_absorbs_ball_zero hr
  /-
    🎉 no goals
  -/


theorem WithSeminorms.image_isVonNBounded_iff_finset_seminorm_bounded (f : G → E) {s : Set G}
    (hp : WithSeminorms p) :
    Bornology.IsVonNBounded 𝕜 (f '' s) ↔
      ∀ I : Finset ι, ∃ r > 0, ∀ x ∈ s, I.sup p (f x) < r := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    G : Type u_7
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    f : G → E
    s : Set G
    hp : WithSeminorms p
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.image f s)) (∀ (I : Finset ι), Exists fu …
  -/
  simp_rw [hp.isVonNBounded_iff_finset_seminorm_bounded, Set.forall_mem_image]
  /-
    🎉 no goals
  -/


theorem WithSeminorms.isVonNBounded_iff_seminorm_bounded {s : Set E} (hp : WithSeminorms p) :
    Bornology.IsVonNBounded 𝕜 s ↔ ∀ i : ι, ∃ r > 0, ∀ x ∈ s, p i x < r := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 s) (∀ (i : ι), Exists fun r => And (GT.gt r 0 …
  -/
  rw [hp.isVonNBounded_iff_finset_seminorm_bounded]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ Iff (∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membershi …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      ⊢ (∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.me …
    -/
  · intro hI i
    /-
      case mp
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      hI : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership. …
      i : ι
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((p i …
    -/
    convert hI {i}
    /-
      case h.e'_2.h.h.e'_2.h.h'.h.e'_3.h.e'_5
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      hI : ∀ (I : Finset ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership. …
      i : ι
      x✝ : Real
      a✝¹ : E
      a✝ : Membership.mem s a✝¹
      ⊢ Eq (p i) ((Singleton.singleton i).sup p)
    -/
    rw [Finset.sup_singleton]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    ⊢ (∀ (i : ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → …
  -/
  intro hi I
  /-
    case mpr
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    hi : ∀ (i : ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x …
    I : Finset ι
    ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
  -/
  by_cases hI : I.Nonempty
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      hi : ∀ (i : ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x …
      I : Finset ι
      hI : I.Nonempty
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
  · choose r hr h using hi
    have h' : 0 < I.sup' hI r := by
      rcases hI with ⟨i, hi⟩
      exact lt_of_lt_of_le (hr i) (Finset.le_sup' r hi)
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      hI : I.Nonempty
      r : ι → Real
      hr : ∀ (i : ι), GT.gt (r i) 0
      h : ∀ (i : ι) (x : E), Membership.mem s x → LT.lt ((p i) x) (r i)
      h' : LT.lt 0 (I.sup' hI r)
      ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt ((I.s …
    -/
    refine ⟨I.sup' hI r, h', fun x hx => finset_sup_apply_lt h' fun i hi => ?_⟩
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      hI : I.Nonempty
      r : ι → Real
      hr : ∀ (i : ι), GT.gt (r i) 0
      h : ∀ (i : ι) (x : E), Membership.mem s x → LT.lt ((p i) x) (r i)
      h' : LT.lt 0 (I.sup' hI r)
      x : E
      hx : Membership.mem s x
      i : ι
      hi : Membership.mem I i
      ⊢ LT.lt ((p i) x) (I.sup' hI r)
    -/
    refine lt_of_lt_of_le (h i x hx) ?_
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      hI : I.Nonempty
      r : ι → Real
      hr : ∀ (i : ι), GT.gt (r i) 0
      h : ∀ (i : ι) (x : E), Membership.mem s x → LT.lt ((p i) x) (r i)
      h' : LT.lt 0 (I.sup' hI r)
      x : E
      hx : Membership.mem s x
      i : ι
      hi : Membership.mem I i
      ⊢ LE.le (r i) (I.sup' hI r)
    -/
    simp only [Finset.le_sup'_iff, exists_prop]
    /-
      case pos
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : AddCommGroup E
      inst✝² : Module 𝕜 E
      inst✝¹ : Nonempty ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      s : Set E
      hp : WithSeminorms p
      I : Finset ι
      hI : I.Nonempty
      r : ι → Real
      hr : ∀ (i : ι), GT.gt (r i) 0
      h : ∀ (i : ι) (x : E), Membership.mem s x → LT.lt ((p i) x) (r i)
      h' : LT.lt 0 (I.sup' hI r)
      x : E
      hx : Membership.mem s x
      i : ι
      hi : Membership.mem I i
      ⊢ Exists fun b => And (Membership.mem I b) (LE.le (r i) (r b))
    -/
    exact ⟨i, hi, (Eq.refl _).le⟩
    /-
      🎉 no goals
    -/
  simp only [Finset.not_nonempty_iff_eq_empty.mp hI, Finset.sup_empty, coe_bot, Pi.zero_apply,
    exists_prop]
  /-
    case neg
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    s : Set E
    hp : WithSeminorms p
    hi : ∀ (i : ι), Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x …
    I : Finset ι
    hI : Not I.Nonempty
    ⊢ Exists fun r => And (GT.gt r 0) (∀ (x : E), Membership.mem s x → LT.lt 0 r)
  -/
  exact ⟨1, zero_lt_one, fun _ _ => zero_lt_one⟩
  /-
    🎉 no goals
  -/


theorem WithSeminorms.image_isVonNBounded_iff_seminorm_bounded (f : G → E) {s : Set G}
    (hp : WithSeminorms p) :
    Bornology.IsVonNBounded 𝕜 (f '' s) ↔ ∀ i : ι, ∃ r > 0, ∀ x ∈ s, p i (f x) < r := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    G : Type u_7
    ι : Type u_8
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    inst✝¹ : Nonempty ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    f : G → E
    s : Set G
    hp : WithSeminorms p
    ⊢ Iff (Bornology.IsVonNBounded 𝕜 (Set.image f s)) (∀ (i : ι), Exists fun r =>  …
  -/
  simp_rw [hp.isVonNBounded_iff_seminorm_bounded, Set.forall_mem_image]
  /-
    🎉 no goals
  -/


theorem continuous_of_continuous_comp {q : SeminormFamily 𝕝₂ F ι'} [TopologicalSpace E]
    [TopologicalAddGroup E] [TopologicalSpace F] (hq : WithSeminorms q)
    (f : E →ₛₗ[τ₁₂] F) (hf : ∀ i, Continuous ((q i).comp f)) : Continuous f := by
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : NormedField 𝕝
    inst✝⁸ : Module 𝕝 E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι'
    q : SeminormFamily 𝕝₂ F ι'
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι'), Continuous ⇑((q i).comp f)
    ⊢ Continuous ⇑f
  -/
  have : TopologicalAddGroup F := hq.topologicalAddGroup
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : NormedField 𝕝
    inst✝⁸ : Module 𝕝 E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι'
    q : SeminormFamily 𝕝₂ F ι'
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι'), Continuous ⇑((q i).comp f)
    this : TopologicalAddGroup F
    ⊢ Continuous ⇑f
  -/
  refine continuous_of_continuousAt_zero f ?_
  simp_rw [ContinuousAt, f.map_zero, q.withSeminorms_iff_nhds_eq_iInf.mp hq, Filter.tendsto_iInf,
    Filter.tendsto_comap_iff]
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : NormedField 𝕝
    inst✝⁸ : Module 𝕝 E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι'
    q : SeminormFamily 𝕝₂ F ι'
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι'), Continuous ⇑((q i).comp f)
    this : TopologicalAddGroup F
    ⊢ ∀ (i : ι'), Filter.Tendsto (Function.comp ⇑(q i) ⇑f) (nhds 0) (nhds 0)
  -/
  intro i
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : NormedField 𝕝
    inst✝⁸ : Module 𝕝 E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι'
    q : SeminormFamily 𝕝₂ F ι'
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι'), Continuous ⇑((q i).comp f)
    this : TopologicalAddGroup F
    i : ι'
    ⊢ Filter.Tendsto (Function.comp ⇑(q i) ⇑f) (nhds 0) (nhds 0)
  -/
  convert (hf i).continuousAt.tendsto
  /-
    case h.e'_5.h.e'_3
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : NormedField 𝕝
    inst✝⁸ : Module 𝕝 E
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι'
    q : SeminormFamily 𝕝₂ F ι'
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι'), Continuous ⇑((q i).comp f)
    this : TopologicalAddGroup F
    i : ι'
    ⊢ Eq 0 (((q i).comp f) 0)
  -/
  exact (map_zero _).symm
  /-
    🎉 no goals
  -/


theorem continuous_iff_continuous_comp {q : SeminormFamily 𝕜₂ F ι'} [TopologicalSpace E]
    [TopologicalAddGroup E] [TopologicalSpace F] (hq : WithSeminorms q) (f : E →ₛₗ[σ₁₂] F) :
    Continuous f ↔ ∀ i, Continuous ((q i).comp f) :=
    -- Porting note: if we *don't* use dot notation for `Continuous.comp`, Lean tries to show
    -- continuity of `((q i).comp f) ∘ id` because it doesn't see that `((q i).comp f)` is
    -- actually a composition of functions.
  ⟨fun h i => (hq.continuous_seminorm i).comp h, continuous_of_continuous_comp hq f⟩


theorem continuous_from_bounded {p : SeminormFamily 𝕝 E ι} {q : SeminormFamily 𝕝₂ F ι'}
    {_ : TopologicalSpace E} (hp : WithSeminorms p) {_ : TopologicalSpace F} (hq : WithSeminorms q)
    (f : E →ₛₗ[τ₁₂] F) (hf : Seminorm.IsBounded p q f) : Continuous f := by
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    ⊢ Continuous ⇑f
  -/
  have : TopologicalAddGroup E := hp.topologicalAddGroup
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    ⊢ Continuous ⇑f
  -/
  refine continuous_of_continuous_comp hq _ fun i => ?_
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    ⊢ Continuous ⇑((q i).comp f)
  -/
  rcases hf i with ⟨s, C, hC⟩
  /-
    case intro.intro
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    s : Finset ι
    C : NNReal
    hC : LE.le ((q i).comp f) (HSMul.hSMul C (s.sup p))
    ⊢ Continuous ⇑((q i).comp f)
  -/
  rw [← Seminorm.finset_sup_smul] at hC
  -- Note: we deduce continuouty of `s.sup (C • p)` from that of `∑ i ∈ s, C • p i`.
  -- The reason is that there is no `continuous_finset_sup`, and even if it were we couldn't
  -- really use it since `ℝ` is not an `OrderBot`.
  /-
    case intro.intro
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    s : Finset ι
    C : NNReal
    hC : LE.le ((q i).comp f) (s.sup (HSMul.hSMul C p))
    ⊢ Continuous ⇑((q i).comp f)
  -/
  refine Seminorm.continuous_of_le ?_ (hC.trans <| Seminorm.finset_sup_le_sum _ _)
  /-
    case intro.intro
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    s : Finset ι
    C : NNReal
    hC : LE.le ((q i).comp f) (s.sup (HSMul.hSMul C p))
    ⊢ Continuous ⇑(s.sum fun i => HSMul.hSMul C p i)
  -/
  change Continuous (fun x ↦ Seminorm.coeFnAddMonoidHom _ _ (∑ i ∈ s, C • p i) x)
  /-
    case intro.intro
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    s : Finset ι
    C : NNReal
    hC : LE.le ((q i).comp f) (s.sup (HSMul.hSMul C p))
    ⊢ Continuous fun x => (Seminorm.coeFnAddMonoidHom 𝕝 E) (s.sum fun i => HSMul.h …
  -/
  simp_rw [map_sum, Finset.sum_apply]
  /-
    case intro.intro
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    ι' : Type u_9
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : AddCommGroup F
    inst✝⁴ : NormedField 𝕝₂
    inst✝³ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝² : RingHomIsometric τ₁₂
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    p : SeminormFamily 𝕝 E ι
    q : SeminormFamily 𝕝₂ F ι'
    x✝¹ : TopologicalSpace E
    hp : WithSeminorms p
    x✝ : TopologicalSpace F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p q f
    this : TopologicalAddGroup E
    i : ι'
    s : Finset ι
    C : NNReal
    hC : LE.le ((q i).comp f) (s.sup (HSMul.hSMul C p))
    ⊢ Continuous fun x => s.sum fun c => (Seminorm.coeFnAddMonoidHom 𝕝 E) (HSMul.h …
  -/
  exact (continuous_finset_sum _ fun i _ ↦ (hp.continuous_seminorm i).const_smul (C : ℝ))
  /-
    🎉 no goals
  -/


theorem cont_withSeminorms_normedSpace (F) [SeminormedAddCommGroup F] [NormedSpace 𝕝₂ F]
    [TopologicalSpace E] {p : ι → Seminorm 𝕝 E} (hp : WithSeminorms p)
    (f : E →ₛₗ[τ₁₂] F) (hf : ∃ (s : Finset ι) (C : ℝ≥0), (normSeminorm 𝕝₂ F).comp f ≤ C • s.sup p) :
    Continuous f := by
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    ι : Type u_8
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : NormedField 𝕝₂
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι
    F : Type u_10
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕝₂ F
    inst✝ : TopologicalSpace E
    p : ι → Seminorm 𝕝 E
    hp : WithSeminorms p
    f : LinearMap τ₁₂ E F
    hf : Exists fun s => Exists fun C => LE.le ((normSeminorm 𝕝₂ F).comp f) (HSMul …
    ⊢ Continuous ⇑f
  -/
  rw [← Seminorm.isBounded_const (Fin 1)] at hf
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    E : Type u_5
    ι : Type u_8
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : NormedField 𝕝
    inst✝⁶ : Module 𝕝 E
    inst✝⁵ : NormedField 𝕝₂
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι
    F : Type u_10
    inst✝² : SeminormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕝₂ F
    inst✝ : TopologicalSpace E
    p : ι → Seminorm 𝕝 E
    hp : WithSeminorms p
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded p (fun x => normSeminorm 𝕝₂ F) f
    ⊢ Continuous ⇑f
  -/
  exact continuous_from_bounded hp (norm_withSeminorms 𝕝₂ F) f hf
  /-
    🎉 no goals
  -/


theorem cont_normedSpace_to_withSeminorms (E) [SeminormedAddCommGroup E] [NormedSpace 𝕝 E]
    [TopologicalSpace F] {q : ι → Seminorm 𝕝₂ F} (hq : WithSeminorms q)
    (f : E →ₛₗ[τ₁₂] F) (hf : ∀ i : ι, ∃ C : ℝ≥0, (q i).comp f ≤ C • normSeminorm 𝕝 E) :
    Continuous f := by
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    F : Type u_6
    ι : Type u_8
    inst✝⁸ : NormedField 𝕝
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι
    E : Type u_10
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕝 E
    inst✝ : TopologicalSpace F
    q : ι → Seminorm 𝕝₂ F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : ∀ (i : ι), Exists fun C => LE.le ((q i).comp f) (HSMul.hSMul C (normSemin …
    ⊢ Continuous ⇑f
  -/
  rw [← Seminorm.const_isBounded (Fin 1)] at hf
  /-
    𝕝 : Type u_3
    𝕝₂ : Type u_4
    F : Type u_6
    ι : Type u_8
    inst✝⁸ : NormedField 𝕝
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : NormedField 𝕝₂
    inst✝⁵ : Module 𝕝₂ F
    τ₁₂ : RingHom 𝕝 𝕝₂
    inst✝⁴ : RingHomIsometric τ₁₂
    inst✝³ : Nonempty ι
    E : Type u_10
    inst✝² : SeminormedAddCommGroup E
    inst✝¹ : NormedSpace 𝕝 E
    inst✝ : TopologicalSpace F
    q : ι → Seminorm 𝕝₂ F
    hq : WithSeminorms q
    f : LinearMap τ₁₂ E F
    hf : Seminorm.IsBounded (fun x => normSeminorm 𝕝 E) q f
    ⊢ Continuous ⇑f
  -/
  exact continuous_from_bounded (norm_withSeminorms 𝕝 E) hq f hf
  /-
    🎉 no goals
  -/


/-- Let `E` and `F` be two topological vector spaces over a `NontriviallyNormedField`, and assume
that the topology of `F` is generated by some family of seminorms `q`. For a family `f` of linear
maps from `E` to `F`, the following are equivalent:
* `f` is equicontinuous at `0`.
* `f` is equicontinuous.
* `f` is uniformly equicontinuous.
* For each `q i`, the family of seminorms `k ↦ (q i) ∘ (f k)` is bounded by some continuous
  seminorm `p` on `E`.
* For each `q i`, the seminorm `⊔ k, (q i) ∘ (f k)` is well-defined and continuous.

In particular, if you can determine all continuous seminorms on `E`, that gives you a complete
characterization of equicontinuity for linear maps from `E` to `F`. For example `E` and `F` are
both normed spaces, you get `NormedSpace.equicontinuous_TFAE`. -/
protected theorem _root_.WithSeminorms.equicontinuous_TFAE {κ : Type*}
    {q : SeminormFamily 𝕜₂ F ι'} [UniformSpace E] [UniformAddGroup E] [u : UniformSpace F]
    [hu : UniformAddGroup F] (hq : WithSeminorms q) [ContinuousSMul 𝕜 E]
    (f : κ → E →ₛₗ[σ₁₂] F) : TFAE
    [ EquicontinuousAt ((↑) ∘ f) 0,
      Equicontinuous ((↑) ∘ f),
      UniformEquicontinuous ((↑) ∘ f),
      ∀ i, ∃ p : Seminorm 𝕜 E, Continuous p ∧ ∀ k, (q i).comp (f k) ≤ p,
      ∀ i, BddAbove (range fun k ↦ (q i).comp (f k)) ∧ Continuous (⨆ k, (q i).comp (f k)) ] := by
  -- We start by reducing to the case where the target is a seminormed space
  rw [q.withSeminorms_iff_uniformSpace_eq_iInf.mp hq, uniformEquicontinuous_iInf_rng,
      equicontinuous_iInf_rng, equicontinuousAt_iInf_rng]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    u : UniformSpace F
    hu : UniformAddGroup F
    hq : WithSeminorms q
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    ⊢ (List.cons (∀ (k : ι'), EquicontinuousAt (Function.comp DFunLike.coe f) 0) ( …
  -/
  refine forall_tfae [_, _, _, _, _] fun i ↦ ?_
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    u : UniformSpace F
    hu : UniformAddGroup F
    hq : WithSeminorms q
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    ⊢ (List.map (fun p => p i) (List.cons (fun k => EquicontinuousAt (Function.com …
  -/
  let _ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    u : UniformSpace F
    hu : UniformAddGroup F
    hq : WithSeminorms q
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    ⊢ (List.map (fun p => p i) (List.cons (fun k => EquicontinuousAt (Function.com …
  -/
  clear u hu hq
  -- Now we can prove the equivalence in this setting
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    ⊢ (List.map (fun p => p i) (List.cons (fun k => EquicontinuousAt (Function.com …
  -/
  simp only [List.map]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 1 → 3 := uniformEquicontinuous_of_equicontinuousAt_zero f
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 3 → 2 := UniformEquicontinuous.equicontinuous
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    tfae_3_to_2 : UniformEquicontinuous (Function.comp DFunLike.coe f) → Equiconti …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 2 → 1 := fun H ↦ H 0
  tfae_have 3 → 5
  | H => by
    have : ∀ᶠ x in 𝓝 0, ∀ k, q i (f k x) ≤ 1 := by
      filter_upwards [Metric.equicontinuousAt_iff_right.mp (H.equicontinuous 0) 1 one_pos]
        with x hx k
      simpa using (hx k).le
    have bdd : BddAbove (range fun k ↦ (q i).comp (f k)) :=
      Seminorm.bddAbove_of_absorbent (absorbent_nhds_zero this)
        (fun x hx ↦ ⟨1, forall_mem_range.mpr hx⟩)
    rw [← Seminorm.coe_iSup_eq bdd]
    refine ⟨bdd, Seminorm.continuous' (r := 1) ?_⟩
    filter_upwards [this] with x hx
    simpa only [closedBall_iSup bdd _ one_pos, mem_iInter, mem_closedBall_zero] using hx
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    tfae_3_to_2 : UniformEquicontinuous (Function.comp DFunLike.coe f) → Equiconti …
    tfae_2_to_1 : Equicontinuous (Function.comp DFunLike.coe f) → EquicontinuousAt …
    tfae_3_to_5 : UniformEquicontinuous (Function.comp DFunLike.coe f) → And (BddA …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_have 5 → 4 := fun H ↦ ⟨⨆ k, (q i).comp (f k), Seminorm.coe_iSup_eq H.1 ▸ H.2, le_ciSup H.1⟩
  tfae_have 4 → 1 -- This would work over any `NormedField`
  | ⟨p, hp, hfp⟩ =>
    Metric.equicontinuousAt_of_continuity_modulus p (map_zero p ▸ hp.tendsto 0) _ <|
      Eventually.of_forall fun x k ↦ by simpa using hfp k x
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹⁰ : NontriviallyNormedField 𝕜
    inst✝⁹ : AddCommGroup E
    inst✝⁸ : Module 𝕜 E
    inst✝⁷ : NontriviallyNormedField 𝕜₂
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁴ : RingHomIsometric σ₁₂
    inst✝³ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝² : UniformSpace E
    inst✝¹ : UniformAddGroup E
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    i : ι'
    x✝ : SeminormedAddCommGroup F := (q i).toSeminormedAddCommGroup
    tfae_1_to_3 : EquicontinuousAt (Function.comp DFunLike.coe f) 0 → UniformEquic …
    tfae_3_to_2 : UniformEquicontinuous (Function.comp DFunLike.coe f) → Equiconti …
    tfae_2_to_1 : Equicontinuous (Function.comp DFunLike.coe f) → EquicontinuousAt …
    tfae_3_to_5 : UniformEquicontinuous (Function.comp DFunLike.coe f) → And (BddA …
    tfae_5_to_4 : And (BddAbove (Set.range fun k => (q i).comp (f k))) (Continuous …
    tfae_4_to_1 : (Exists fun p => And (Continuous ⇑p) (∀ (k : κ), LE.le ((q i).co …
    ⊢ (List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.cons (E …
  -/
  tfae_finish
  /-
    🎉 no goals
  -/


theorem _root_.WithSeminorms.uniformEquicontinuous_iff_exists_continuous_seminorm {κ : Type*}
    {q : SeminormFamily 𝕜₂ F ι'} [UniformSpace E] [UniformAddGroup E] [u : UniformSpace F]
    [UniformAddGroup F] (hq : WithSeminorms q) [ContinuousSMul 𝕜 E]
    (f : κ → E →ₛₗ[σ₁₂] F) :
    UniformEquicontinuous ((↑) ∘ f) ↔
    ∀ i, ∃ p : Seminorm 𝕜 E, Continuous p ∧ ∀ k, (q i).comp (f k) ≤ p :=
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜 E
    inst✝⁸ : NontriviallyNormedField 𝕜₂
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁵ : RingHomIsometric σ₁₂
    inst✝⁴ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝³ : UniformSpace E
    inst✝² : UniformAddGroup E
    u : UniformSpace F
    inst✝¹ : UniformAddGroup F
    hq : WithSeminorms q
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    ⊢ Eq ((List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.con …
  -/
  /-
    🎉 no goals
  -/
  (hq.equicontinuous_TFAE f).out 2 3
  /-
    🎉 no goals
  -/


theorem _root_.WithSeminorms.uniformEquicontinuous_iff_bddAbove_and_continuous_iSup {κ : Type*}
    {q : SeminormFamily 𝕜₂ F ι'} [UniformSpace E] [UniformAddGroup E] [u : UniformSpace F]
    [UniformAddGroup F] (hq : WithSeminorms q) [ContinuousSMul 𝕜 E]
    (f : κ → E →ₛₗ[σ₁₂] F) :
    UniformEquicontinuous ((↑) ∘ f) ↔ ∀ i,
    BddAbove (range fun k ↦ (q i).comp (f k)) ∧
      Continuous (⨆ k, (q i).comp (f k)) :=
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι' : Type u_9
    inst✝¹¹ : NontriviallyNormedField 𝕜
    inst✝¹⁰ : AddCommGroup E
    inst✝⁹ : Module 𝕜 E
    inst✝⁸ : NontriviallyNormedField 𝕜₂
    inst✝⁷ : AddCommGroup F
    inst✝⁶ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝⁵ : RingHomIsometric σ₁₂
    inst✝⁴ : Nonempty ι'
    κ : Type u_10
    q : SeminormFamily 𝕜₂ F ι'
    inst✝³ : UniformSpace E
    inst✝² : UniformAddGroup E
    u : UniformSpace F
    inst✝¹ : UniformAddGroup F
    hq : WithSeminorms q
    inst✝ : ContinuousSMul 𝕜 E
    f : κ → LinearMap σ₁₂ E F
    ⊢ Eq ((List.cons (EquicontinuousAt (Function.comp DFunLike.coe f) 0) (List.con …
  -/
  /-
    🎉 no goals
  -/
  (hq.equicontinuous_TFAE f).out 2 4
  /-
    🎉 no goals
  -/


/-- Two families of seminorms `p` and `q` on the same space generate the same topology
if each `p i` is bounded by some `C • Finset.sup s q` and vice-versa.

We formulate these boundedness assumptions as `Seminorm.IsBounded q p LinearMap.id` (and
vice-versa) to reuse the API. Furthermore, we don't actually state it as an equality of topologies
but as a way to deduce `WithSeminorms q` from `WithSeminorms p`, since this should be more
useful in practice. -/
protected theorem congr {p : SeminormFamily 𝕜 E ι} {q : SeminormFamily 𝕜 E ι'}
    [t : TopologicalSpace E] (hp : WithSeminorms p) (hpq : Seminorm.IsBounded p q LinearMap.id)
    (hqp : Seminorm.IsBounded q p LinearMap.id) : WithSeminorms q := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    q : SeminormFamily 𝕜 E ι'
    t : TopologicalSpace E
    hp : WithSeminorms p
    hpq : Seminorm.IsBounded p q LinearMap.id
    hqp : Seminorm.IsBounded q p LinearMap.id
    ⊢ WithSeminorms q
  -/
  constructor
  /-
    case topology_eq_withSeminorms
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    q : SeminormFamily 𝕜 E ι'
    t : TopologicalSpace E
    hp : WithSeminorms p
    hpq : Seminorm.IsBounded p q LinearMap.id
    hqp : Seminorm.IsBounded q p LinearMap.id
    ⊢ Eq t q.moduleFilterBasis.topology
  -/
  rw [hp.topology_eq_withSeminorms]
  /-
    case topology_eq_withSeminorms
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    q : SeminormFamily 𝕜 E ι'
    t : TopologicalSpace E
    hp : WithSeminorms p
    hpq : Seminorm.IsBounded p q LinearMap.id
    hqp : Seminorm.IsBounded q p LinearMap.id
    ⊢ Eq p.moduleFilterBasis.topology q.moduleFilterBasis.topology
  -/
  clear hp t
  /-
    case topology_eq_withSeminorms
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    q : SeminormFamily 𝕜 E ι'
    hpq : Seminorm.IsBounded p q LinearMap.id
    hqp : Seminorm.IsBounded q p LinearMap.id
    ⊢ Eq p.moduleFilterBasis.topology q.moduleFilterBasis.topology
  -/
  refine le_antisymm ?_ ?_ <;>
  /-
    case topology_eq_withSeminorms.refine_1
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    q : SeminormFamily 𝕜 E ι'
    hpq : Seminorm.IsBounded p q LinearMap.id
    hqp : Seminorm.IsBounded q p LinearMap.id
    ⊢ LE.le p.moduleFilterBasis.topology q.moduleFilterBasis.topology
  -/
  rw [← continuous_id_iff_le] <;>
  refine continuous_from_bounded (.mk (topology := _) rfl) (.mk (topology := _) rfl)
    LinearMap.id (by assumption)


protected theorem finset_sups {p : SeminormFamily 𝕜 E ι} [TopologicalSpace E]
    (hp : WithSeminorms p) : WithSeminorms (fun s : Finset ι ↦ s.sup p) := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : Nonempty ι
    inst✝³ : NormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    ⊢ WithSeminorms fun s => s.sup p
  -/
  refine hp.congr ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      ⊢ Seminorm.IsBounded p (fun s => s.sup p) LinearMap.id
    -/
  · intro s
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      s : Finset ι
      ⊢ Exists fun s_1 => Exists fun C => LE.le (((fun s => s.sup p) s).comp LinearM …
    -/
    refine ⟨s, 1, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      s : Finset ι
      ⊢ LE.le (((fun s => s.sup p) s).comp LinearMap.id) (HSMul.hSMul 1 (s.sup p))
    -/
    rw [one_smul]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      s : Finset ι
      ⊢ LE.le (((fun s => s.sup p) s).comp LinearMap.id) (s.sup p)
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      ⊢ Seminorm.IsBounded (fun s => s.sup p) p LinearMap.id
    -/
  · intro i
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ Exists fun s => Exists fun C => LE.le ((p i).comp LinearMap.id) (HSMul.hSMul …
    -/
    refine ⟨{{i}}, 1, ?_⟩
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le ((p i).comp LinearMap.id) (HSMul.hSMul 1 ((Singleton.singleton (Single …
    -/
    rw [Finset.sup_singleton, Finset.sup_singleton, one_smul]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁴ : Nonempty ι
      inst✝³ : NormedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le ((p i).comp LinearMap.id) (p i)
    -/
    rfl
    /-
      🎉 no goals
    -/


protected theorem partial_sups [Preorder ι] [LocallyFiniteOrderBot ι] {p : SeminormFamily 𝕜 E ι}
    [TopologicalSpace E] (hp : WithSeminorms p) : WithSeminorms (fun i ↦ (Finset.Iic i).sup p) := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁶ : Nonempty ι
    inst✝⁵ : NormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Preorder ι
    inst✝¹ : LocallyFiniteOrderBot ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    ⊢ WithSeminorms fun i => (Finset.Iic i).sup p
  -/
  refine hp.congr ?_ ?_
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      ⊢ Seminorm.IsBounded p (fun i => (Finset.Iic i).sup p) LinearMap.id
    -/
  · intro i
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ Exists fun s => Exists fun C => LE.le (((fun i => (Finset.Iic i).sup p) i).c …
    -/
    refine ⟨Finset.Iic i, 1, ?_⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le (((fun i => (Finset.Iic i).sup p) i).comp LinearMap.id) (HSMul.hSMul 1 …
    -/
    rw [one_smul]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le (((fun i => (Finset.Iic i).sup p) i).comp LinearMap.id) ((Finset.Iic i …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      ⊢ Seminorm.IsBounded (fun i => (Finset.Iic i).sup p) p LinearMap.id
    -/
  · intro i
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ Exists fun s => Exists fun C => LE.le ((p i).comp LinearMap.id) (HSMul.hSMul …
    -/
    refine ⟨{i}, 1, ?_⟩
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le ((p i).comp LinearMap.id) (HSMul.hSMul 1 ((Singleton.singleton i).sup  …
    -/
    rw [Finset.sup_singleton, one_smul]
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁶ : Nonempty ι
      inst✝⁵ : NormedField 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Preorder ι
      inst✝¹ : LocallyFiniteOrderBot ι
      p : SeminormFamily 𝕜 E ι
      inst✝ : TopologicalSpace E
      hp : WithSeminorms p
      i : ι
      ⊢ LE.le ((p i).comp LinearMap.id) ((Finset.Iic i).sup p)
    -/
    exact (Finset.le_sup (Finset.mem_Iic.mpr le_rfl) : p i ≤ (Finset.Iic i).sup p)
    /-
      🎉 no goals
    -/


protected theorem congr_equiv {p : SeminormFamily 𝕜 E ι} [t : TopologicalSpace E]
    (hp : WithSeminorms p) (e : ι' ≃ ι) : WithSeminorms (p ∘ e) := by
  refine hp.congr ?_ ?_ <;>
  intro i <;>
  [use {e i}, 1; use {e.symm i}, 1] <;>
  /-
    case h
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    ι' : Type u_9
    inst✝⁴ : Nonempty ι
    inst✝³ : Nonempty ι'
    inst✝² : NormedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    t : TopologicalSpace E
    hp : WithSeminorms p
    e : Equiv ι' ι
    i : ι'
    ⊢ LE.le ((Function.comp p (⇑e) i).comp LinearMap.id) (HSMul.hSMul 1 ((Singleto …
  -/
  /-
    🎉 no goals
  -/
  simp
  /-
    🎉 no goals
  -/


/-- In a semi-`NormedSpace`, a continuous seminorm is zero on elements of norm `0`. -/
lemma map_eq_zero_of_norm_zero (q : Seminorm 𝕜 F)
    (hq : Continuous q) {x : F} (hx : ‖x‖ = 0) : q x = 0 :=
  (map_zero q) ▸
    ((specializes_iff_mem_closure.mpr <| mem_closure_zero_iff_norm.mpr hx).map hq).eq.symm


/-- Let `F` be a semi-`NormedSpace` over a `NontriviallyNormedField`, and let `q` be a
seminorm on `F`. If `q` is continuous, then it is uniformly controlled by the norm, that is there
is some `C > 0` such that `∀ x, q x ≤ C * ‖x‖`.
The continuity ensures boundedness on a ball of some radius `ε`. The nontriviality of the
norm is then used to rescale any element into an element of norm in `[ε/C, ε[`, thus with a
controlled image by `q`. The control of `q` at the original element follows by rescaling. -/
lemma bound_of_continuous_normedSpace (q : Seminorm 𝕜 F)
    (hq : Continuous q) : ∃ C, 0 < C ∧ (∀ x : F, q x ≤ C * ‖x‖) := by
  /-
    𝕜 : Type u_1
    F : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    q : Seminorm 𝕜 F
    hq : Continuous ⇑q
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : F), LE.le (q x) (HMul.hMul C (Norm.n …
  -/
  have hq' : Tendsto q (𝓝 0) (𝓝 0) := map_zero q ▸ hq.tendsto 0
  rcases NormedAddCommGroup.nhds_zero_basis_norm_lt.mem_iff.mp (hq' <| Iio_mem_nhds one_pos)
    with ⟨ε, ε_pos, hε⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    F : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    q : Seminorm 𝕜 F
    hq : Continuous ⇑q
    hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
    ε : Real
    ε_pos : LT.lt 0 ε
    hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : F), LE.le (q x) (HMul.hMul C (Norm.n …
  -/
  rcases NormedField.exists_one_lt_norm 𝕜 with ⟨c, hc⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    F : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    q : Seminorm 𝕜 F
    hq : Continuous ⇑q
    hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
    ε : Real
    ε_pos : LT.lt 0 ε
    hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : F), LE.le (q x) (HMul.hMul C (Norm.n …
  -/
  have : 0 < ‖c‖ / ε := by positivity
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    F : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    q : Seminorm 𝕜 F
    hq : Continuous ⇑q
    hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
    ε : Real
    ε_pos : LT.lt 0 ε
    hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
    ⊢ Exists fun C => And (LT.lt 0 C) (∀ (x : F), LE.le (q x) (HMul.hMul C (Norm.n …
  -/
  refine ⟨‖c‖ / ε, this, fun x ↦ ?_⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    F : Type u_6
    inst✝² : NontriviallyNormedField 𝕜
    inst✝¹ : SeminormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    q : Seminorm 𝕜 F
    hq : Continuous ⇑q
    hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
    ε : Real
    ε_pos : LT.lt 0 ε
    hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
    c : 𝕜
    hc : LT.lt 1 (Norm.norm c)
    this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
    x : F
    ⊢ LE.le (q x) (HMul.hMul (HDiv.hDiv (Norm.norm c) ε) (Norm.norm x))
  -/
  by_cases hx : ‖x‖ = 0
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_6
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      q : Seminorm 𝕜 F
      hq : Continuous ⇑q
      hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
      ε : Real
      ε_pos : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
      x : F
      hx : Eq (Norm.norm x) 0
      ⊢ LE.le (q x) (HMul.hMul (HDiv.hDiv (Norm.norm c) ε) (Norm.norm x))
    -/
  · rw [hx, mul_zero]
    /-
      case pos
      𝕜 : Type u_1
      F : Type u_6
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      q : Seminorm 𝕜 F
      hq : Continuous ⇑q
      hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
      ε : Real
      ε_pos : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
      x : F
      hx : Eq (Norm.norm x) 0
      ⊢ LE.le (q x) 0
    -/
    exact le_of_eq (map_eq_zero_of_norm_zero q hq hx)
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      F : Type u_6
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      q : Seminorm 𝕜 F
      hq : Continuous ⇑q
      hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
      ε : Real
      ε_pos : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
      x : F
      hx : Not (Eq (Norm.norm x) 0)
      ⊢ LE.le (q x) (HMul.hMul (HDiv.hDiv (Norm.norm c) ε) (Norm.norm x))
    -/
  · refine (normSeminorm 𝕜 F).bound_of_shell q ε_pos hc (fun x hle hlt ↦ ?_) hx
    /-
      case neg
      𝕜 : Type u_1
      F : Type u_6
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      q : Seminorm 𝕜 F
      hq : Continuous ⇑q
      hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
      ε : Real
      ε_pos : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
      x✝ : F
      hx : Not (Eq (Norm.norm x✝) 0)
      x : F
      hle : LE.le (HDiv.hDiv ε (Norm.norm c)) ((normSeminorm 𝕜 F) x)
      hlt : LT.lt ((normSeminorm 𝕜 F) x) ε
      ⊢ LE.le (q x) (HMul.hMul (HDiv.hDiv (Norm.norm c) ε) ((normSeminorm 𝕜 F) x))
    -/
    refine (le_of_lt <| show q x < _ from hε hlt).trans ?_
    /-
      case neg
      𝕜 : Type u_1
      F : Type u_6
      inst✝² : NontriviallyNormedField 𝕜
      inst✝¹ : SeminormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      q : Seminorm 𝕜 F
      hq : Continuous ⇑q
      hq' : Filter.Tendsto (⇑q) (nhds 0) (nhds 0)
      ε : Real
      ε_pos : LT.lt 0 ε
      hε : HasSubset.Subset (setOf fun y => LT.lt (Norm.norm y) ε) (Set.preimage (⇑q …
      c : 𝕜
      hc : LT.lt 1 (Norm.norm c)
      this : LT.lt 0 (HDiv.hDiv (Norm.norm c) ε)
      x✝ : F
      hx : Not (Eq (Norm.norm x✝) 0)
      x : F
      hle : LE.le (HDiv.hDiv ε (Norm.norm c)) ((normSeminorm 𝕜 F) x)
      hlt : LT.lt ((normSeminorm 𝕜 F) x) ε
      ⊢ LE.le 1 (HMul.hMul (HDiv.hDiv (Norm.norm c) ε) ((normSeminorm 𝕜 F) x))
    -/
    rwa [← div_le_iff₀' this, one_div_div]
    /-
      🎉 no goals
    -/


/-- Let `E` be a topological vector space (over a `NontriviallyNormedField`) whose topology is
generated by some family of seminorms `p`, and let `q` be a seminorm on `E`. If `q` is continuous,
then it is uniformly controlled by *finitely many* seminorms of `p`, that is there
is some finset `s` of the index set and some `C > 0` such that `q ≤ C • s.sup p`. -/
lemma bound_of_continuous [Nonempty ι] [t : TopologicalSpace E] (hp : WithSeminorms p)
    (q : Seminorm 𝕜 E) (hq : Continuous q) :
    ∃ s : Finset ι, ∃ C : ℝ≥0, C ≠ 0 ∧ q ≤ C • s.sup p := by
  -- The continuity of `q` gives us a finset `s` and a real `ε > 0`
  -- such that `hε : (s.sup p).ball 0 ε ⊆ q.ball 0 1`.
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    hp : WithSeminorms p
    q : Seminorm 𝕜 E
    hq : Continuous ⇑q
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  rcases hp.hasBasis.mem_iff.mp (ball_mem_nhds hq one_pos) with ⟨V, hV, hε⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    hp : WithSeminorms p
    q : Seminorm 𝕜 E
    hq : Continuous ⇑q
    V : Set E
    hV : Membership.mem p.basisSets V
    hε : HasSubset.Subset (id V) (q.ball 0 1)
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  rcases p.basisSets_iff.mp hV with ⟨s, ε, ε_pos, rfl⟩
  -- Now forget that `E` already had a topology and view it as the (semi)normed space
  -- `(E, s.sup p)`.
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    t : TopologicalSpace E
    hp : WithSeminorms p
    q : Seminorm 𝕜 E
    hq : Continuous ⇑q
    s : Finset ι
    ε : Real
    ε_pos : LT.lt 0 ε
    hV : Membership.mem p.basisSets ((s.sup p).ball 0 ε)
    hε : HasSubset.Subset (id ((s.sup p).ball 0 ε)) (q.ball 0 1)
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  clear hp hq t
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    q : Seminorm 𝕜 E
    s : Finset ι
    ε : Real
    ε_pos : LT.lt 0 ε
    hV : Membership.mem p.basisSets ((s.sup p).ball 0 ε)
    hε : HasSubset.Subset (id ((s.sup p).ball 0 ε)) (q.ball 0 1)
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  let _ : SeminormedAddCommGroup E := (s.sup p).toSeminormedAddCommGroup
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    q : Seminorm 𝕜 E
    s : Finset ι
    ε : Real
    ε_pos : LT.lt 0 ε
    hV : Membership.mem p.basisSets ((s.sup p).ball 0 ε)
    hε : HasSubset.Subset (id ((s.sup p).ball 0 ε)) (q.ball 0 1)
    x✝ : SeminormedAddCommGroup E := (s.sup p).toSeminormedAddCommGroup
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  let _ : NormedSpace 𝕜 E := { norm_smul_le := fun a b ↦ le_of_eq (map_smul_eq_mul (s.sup p) a b) }
  -- The inclusion `hε` tells us exactly that `q` is *still* continuous for this new topology
  have : Continuous q :=
    Seminorm.continuous (r := 1) (mem_of_superset (Metric.ball_mem_nhds _ ε_pos) hε)
  -- Hence we can conclude by applying `bound_of_continuous_normedSpace`.
  /-
    case intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    q : Seminorm 𝕜 E
    s : Finset ι
    ε : Real
    ε_pos : LT.lt 0 ε
    hV : Membership.mem p.basisSets ((s.sup p).ball 0 ε)
    hε : HasSubset.Subset (id ((s.sup p).ball 0 ε)) (q.ball 0 1)
    x✝¹ : SeminormedAddCommGroup E := (s.sup p).toSeminormedAddCommGroup
    x✝ : NormedSpace 𝕜 E := NormedSpace.mk ⋯
    this : Continuous ⇑q
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  rcases bound_of_continuous_normedSpace q this with ⟨C, C_pos, hC⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝³ : NontriviallyNormedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    p : SeminormFamily 𝕜 E ι
    inst✝ : Nonempty ι
    q : Seminorm 𝕜 E
    s : Finset ι
    ε : Real
    ε_pos : LT.lt 0 ε
    hV : Membership.mem p.basisSets ((s.sup p).ball 0 ε)
    hε : HasSubset.Subset (id ((s.sup p).ball 0 ε)) (q.ball 0 1)
    x✝¹ : SeminormedAddCommGroup E := (s.sup p).toSeminormedAddCommGroup
    x✝ : NormedSpace 𝕜 E := NormedSpace.mk ⋯
    this : Continuous ⇑q
    C : Real
    C_pos : LT.lt 0 C
    hC : ∀ (x : E), LE.le (q x) (HMul.hMul C (Norm.norm x))
    ⊢ Exists fun s => Exists fun C => And (Ne C 0) (LE.le q (HSMul.hSMul C (s.sup  …
  -/
  exact ⟨s, ⟨C, C_pos.le⟩, fun H ↦ C_pos.ne.symm (congr_arg NNReal.toReal H), hC⟩
  /-
    🎉 no goals
  -/
  -- Note that the key ingredient for this proof is that, by scaling arguments hidden in
  -- `Seminorm.continuous`, we only have to look at the `q`-ball of radius one, and the `s` we get
  -- from that will automatically work for all other radii.


theorem WithSeminorms.toLocallyConvexSpace {p : SeminormFamily 𝕜 E ι} (hp : WithSeminorms p) :
    LocallyConvexSpace ℝ E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁷ : Nonempty ι
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedSpace Real 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    ⊢ LocallyConvexSpace Real E
  -/
  have := hp.topologicalAddGroup
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁷ : Nonempty ι
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : NormedSpace Real 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Module Real E
    inst✝¹ : IsScalarTower Real 𝕜 E
    inst✝ : TopologicalSpace E
    p : SeminormFamily 𝕜 E ι
    hp : WithSeminorms p
    this : TopologicalAddGroup E
    ⊢ LocallyConvexSpace Real E
  -/
  apply ofBasisZero ℝ E id fun s => s ∈ p.basisSets
    /-
      case hbasis
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      ⊢ (nhds 0).HasBasis (fun s => Membership.mem p.basisSets s) id
    -/
  · rw [hp.1, AddGroupFilterBasis.nhds_eq _, AddGroupFilterBasis.N_zero]
    /-
      case hbasis
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      ⊢ AddGroupFilterBasis.toFilterBasis.filter.HasBasis (fun s => Membership.mem p …
    -/
    exact FilterBasis.hasBasis _
    /-
      🎉 no goals
    -/
    /-
      case hconvex
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      ⊢ ∀ (i : Set E), Membership.mem p.basisSets i → Convex Real (id i)
    -/
  · intro s hs
    /-
      case hconvex
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      s : Set E
      hs : Membership.mem p.basisSets s
      ⊢ Convex Real (id s)
    -/
    change s ∈ Set.iUnion _ at hs
    /-
      case hconvex
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      s : Set E
      hs : Membership.mem (Set.iUnion fun s => Set.iUnion fun r => Set.iUnion fun x  …
      ⊢ Convex Real (id s)
    -/
    simp_rw [Set.mem_iUnion, Set.mem_singleton_iff] at hs
    /-
      case hconvex
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      s : Set E
      hs : Exists fun i => Exists fun i_1 => Exists fun h => Eq s ((i.sup p).ball 0  …
      ⊢ Convex Real (id s)
    -/
    rcases hs with ⟨I, r, _, rfl⟩
    /-
      case hconvex.intro.intro.intro
      𝕜 : Type u_1
      E : Type u_5
      ι : Type u_8
      inst✝⁷ : Nonempty ι
      inst✝⁶ : NormedField 𝕜
      inst✝⁵ : NormedSpace Real 𝕜
      inst✝⁴ : AddCommGroup E
      inst✝³ : Module 𝕜 E
      inst✝² : Module Real E
      inst✝¹ : IsScalarTower Real 𝕜 E
      inst✝ : TopologicalSpace E
      p : SeminormFamily 𝕜 E ι
      hp : WithSeminorms p
      this : TopologicalAddGroup E
      I : Finset ι
      r : Real
      w✝ : LT.lt 0 r
      ⊢ Convex Real (id ((I.sup p).ball 0 r))
    -/
    exact convex_ball _ _ _
    /-
      🎉 no goals
    -/


/-- Not an instance since `𝕜` can't be inferred. See `NormedSpace.toLocallyConvexSpace` for a
slightly weaker instance version. -/
theorem NormedSpace.toLocallyConvexSpace' [NormedSpace 𝕜 E] [Module ℝ E] [IsScalarTower ℝ 𝕜 E] :
    LocallyConvexSpace ℝ E :=
  (norm_withSeminorms 𝕜 E).toLocallyConvexSpace


/-- See `NormedSpace.toLocallyConvexSpace'` for a slightly stronger version which is not an
instance. -/
instance NormedSpace.toLocallyConvexSpace [NormedSpace ℝ E] : LocallyConvexSpace ℝ E :=
  NormedSpace.toLocallyConvexSpace' ℝ


/-- The family of seminorms obtained by composing each seminorm by a linear map. -/
def SeminormFamily.comp (q : SeminormFamily 𝕜₂ F ι) (f : E →ₛₗ[σ₁₂] F) : SeminormFamily 𝕜 E ι :=
  fun i => (q i).comp f


theorem SeminormFamily.comp_apply (q : SeminormFamily 𝕜₂ F ι) (i : ι) (f : E →ₛₗ[σ₁₂] F) :
    q.comp f i = (q i).comp f :=
  rfl


theorem SeminormFamily.finset_sup_comp (q : SeminormFamily 𝕜₂ F ι) (s : Finset ι)
    (f : E →ₛₗ[σ₁₂] F) : (s.sup q).comp f = s.sup (q.comp f) := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : NormedField 𝕜₂
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    q : SeminormFamily 𝕜₂ F ι
    s : Finset ι
    f : LinearMap σ₁₂ E F
    ⊢ Eq ((s.sup q).comp f) (s.sup (q.comp f))
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : NormedField 𝕜₂
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    q : SeminormFamily 𝕜₂ F ι
    s : Finset ι
    f : LinearMap σ₁₂ E F
    x : E
    ⊢ Eq (((s.sup q).comp f) x) ((s.sup (q.comp f)) x)
  -/
  rw [Seminorm.comp_apply, Seminorm.finset_sup_apply, Seminorm.finset_sup_apply]
  /-
    case h
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁶ : NormedField 𝕜
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module 𝕜 E
    inst✝³ : NormedField 𝕜₂
    inst✝² : AddCommGroup F
    inst✝¹ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝ : RingHomIsometric σ₁₂
    q : SeminormFamily 𝕜₂ F ι
    s : Finset ι
    f : LinearMap σ₁₂ E F
    x : E
    ⊢ Eq ↑(s.sup fun i => ⟨(q i) (f x), ⋯⟩) ↑(s.sup fun i => ⟨(q.comp f i) x, ⋯⟩)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem LinearMap.withSeminorms_induced [hι : Nonempty ι] {q : SeminormFamily 𝕜₂ F ι}
    (hq : WithSeminorms q) (f : E →ₛₗ[σ₁₂] F) :
    WithSeminorms (topology := induced f inferInstance) (q.comp f) := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    f : LinearMap σ₁₂ E F
    ⊢ WithSeminorms (q.comp f)
  -/
  have := hq.topologicalAddGroup
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    f : LinearMap σ₁₂ E F
    this : TopologicalAddGroup F
    ⊢ WithSeminorms (q.comp f)
  -/
  let _ : TopologicalSpace E := induced f inferInstance
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    f : LinearMap σ₁₂ E F
    this : TopologicalAddGroup F
    x✝ : TopologicalSpace E := TopologicalSpace.induced (⇑f) inferInstance
    ⊢ WithSeminorms (q.comp f)
  -/
  have : TopologicalAddGroup E := topologicalAddGroup_induced f
  rw [(q.comp f).withSeminorms_iff_nhds_eq_iInf, nhds_induced, map_zero,
    q.withSeminorms_iff_nhds_eq_iInf.mp hq, Filter.comap_iInf]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    f : LinearMap σ₁₂ E F
    this✝ : TopologicalAddGroup F
    x✝ : TopologicalSpace E := TopologicalSpace.induced (⇑f) inferInstance
    this : TopologicalAddGroup E
    ⊢ Eq (iInf fun i => Filter.comap (⇑f) (Filter.comap (⇑(q i)) (nhds 0))) (iInf  …
  -/
  refine iInf_congr fun i => ?_
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁷ : NormedField 𝕜
    inst✝⁶ : AddCommGroup E
    inst✝⁵ : Module 𝕜 E
    inst✝⁴ : NormedField 𝕜₂
    inst✝³ : AddCommGroup F
    inst✝² : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝¹ : RingHomIsometric σ₁₂
    inst✝ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    f : LinearMap σ₁₂ E F
    this✝ : TopologicalAddGroup F
    x✝ : TopologicalSpace E := TopologicalSpace.induced (⇑f) inferInstance
    this : TopologicalAddGroup E
    i : ι
    ⊢ Eq (Filter.comap (⇑f) (Filter.comap (⇑(q i)) (nhds 0))) (Filter.comap (⇑(q.c …
  -/
  exact Filter.comap_comap
  /-
    🎉 no goals
  -/


lemma Topology.IsInducing.withSeminorms [hι : Nonempty ι] {q : SeminormFamily 𝕜₂ F ι}
    (hq : WithSeminorms q) [TopologicalSpace E] {f : E →ₛₗ[σ₁₂] F} (hf : IsInducing f) :
    WithSeminorms (q.comp f) := by
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : RingHomIsometric σ₁₂
    inst✝¹ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    inst✝ : TopologicalSpace E
    f : LinearMap σ₁₂ E F
    hf : Topology.IsInducing ⇑f
    ⊢ WithSeminorms (q.comp f)
  -/
  rw [hf.eq_induced]
  /-
    𝕜 : Type u_1
    𝕜₂ : Type u_2
    E : Type u_5
    F : Type u_6
    ι : Type u_8
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : Module 𝕜 E
    inst✝⁵ : NormedField 𝕜₂
    inst✝⁴ : AddCommGroup F
    inst✝³ : Module 𝕜₂ F
    σ₁₂ : RingHom 𝕜 𝕜₂
    inst✝² : RingHomIsometric σ₁₂
    inst✝¹ : TopologicalSpace F
    hι : Nonempty ι
    q : SeminormFamily 𝕜₂ F ι
    hq : WithSeminorms q
    inst✝ : TopologicalSpace E
    f : LinearMap σ₁₂ E F
    hf : Topology.IsInducing ⇑f
    ⊢ WithSeminorms (q.comp f)
  -/
  exact f.withSeminorms_induced hq
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias Inducing.withSeminorms := IsInducing.withSeminorms


/-- (Disjoint) union of seminorm families. -/
protected def SeminormFamily.sigma {κ : ι → Type*} (p : (i : ι) → SeminormFamily 𝕜 E (κ i)) :
    SeminormFamily 𝕜 E ((i : ι) × κ i) :=
  fun ⟨i, k⟩ => p i k


theorem withSeminorms_iInf {κ : ι → Type*} [Nonempty ((i : ι) × κ i)] [∀ i, Nonempty (κ i)]
    {p : (i : ι) → SeminormFamily 𝕜 E (κ i)} {t : ι → TopologicalSpace E}
    (hp : ∀ i, WithSeminorms (topology := t i) (p i)) :
    WithSeminorms (topology := ⨅ i, t i) (SeminormFamily.sigma p) := by
  have : ∀ i, @TopologicalAddGroup E (t i) _ :=
    fun i ↦ @WithSeminorms.topologicalAddGroup _ _ _ _ _ _ _ (t i) _ (hp i)
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    κ : ι → Type u_10
    inst✝¹ : Nonempty (Sigma fun i => κ i)
    inst✝ : ∀ (i : ι), Nonempty (κ i)
    p : (i : ι) → SeminormFamily 𝕜 E (κ i)
    t : ι → TopologicalSpace E
    hp : ∀ (i : ι), WithSeminorms (p i)
    this : ∀ (i : ι), TopologicalAddGroup E
    ⊢ WithSeminorms (SeminormFamily.sigma p)
  -/
  have : @TopologicalAddGroup E (⨅ i, t i) _ := topologicalAddGroup_iInf inferInstance
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    κ : ι → Type u_10
    inst✝¹ : Nonempty (Sigma fun i => κ i)
    inst✝ : ∀ (i : ι), Nonempty (κ i)
    p : (i : ι) → SeminormFamily 𝕜 E (κ i)
    t : ι → TopologicalSpace E
    hp : ∀ (i : ι), WithSeminorms (p i)
    this✝ : ∀ (i : ι), TopologicalAddGroup E
    this : TopologicalAddGroup E
    ⊢ WithSeminorms (SeminormFamily.sigma p)
  -/
  simp_rw [@SeminormFamily.withSeminorms_iff_topologicalSpace_eq_iInf _ _ _ _ _ _ _ (_)] at hp ⊢
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    κ : ι → Type u_10
    inst✝¹ : Nonempty (Sigma fun i => κ i)
    inst✝ : ∀ (i : ι), Nonempty (κ i)
    p : (i : ι) → SeminormFamily 𝕜 E (κ i)
    t : ι → TopologicalSpace E
    this✝ : ∀ (i : ι), TopologicalAddGroup E
    this : TopologicalAddGroup E
    hp : ∀ (i : ι), Eq (t i) (iInf fun i_1 => UniformSpace.toTopologicalSpace)
    ⊢ Eq (iInf fun i => t i) (iInf fun i => UniformSpace.toTopologicalSpace)
  -/
  rw [iInf_sigma]
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁴ : NormedField 𝕜
    inst✝³ : AddCommGroup E
    inst✝² : Module 𝕜 E
    κ : ι → Type u_10
    inst✝¹ : Nonempty (Sigma fun i => κ i)
    inst✝ : ∀ (i : ι), Nonempty (κ i)
    p : (i : ι) → SeminormFamily 𝕜 E (κ i)
    t : ι → TopologicalSpace E
    this✝ : ∀ (i : ι), TopologicalAddGroup E
    this : TopologicalAddGroup E
    hp : ∀ (i : ι), Eq (t i) (iInf fun i_1 => UniformSpace.toTopologicalSpace)
    ⊢ Eq (iInf fun i => t i) (iInf fun i => iInf fun j => UniformSpace.toTopologic …
  -/
  exact iInf_congr hp
  /-
    🎉 no goals
  -/


theorem withSeminorms_pi {κ : ι → Type*} {E : ι → Type*}
    [∀ i, AddCommGroup (E i)] [∀ i, Module 𝕜 (E i)] [∀ i, TopologicalSpace (E i)]
    [Nonempty ((i : ι) × κ i)] [∀ i, Nonempty (κ i)] {p : (i : ι) → SeminormFamily 𝕜 (E i) (κ i)}
    (hp : ∀ i, WithSeminorms (p i)) :
    WithSeminorms (SeminormFamily.sigma (fun i ↦ (p i).comp (LinearMap.proj i))) :=
  withSeminorms_iInf fun i ↦ (LinearMap.proj i).withSeminorms_induced (hp i)


/-- If the topology of a space is induced by a countable family of seminorms, then the topology
is first countable. -/
theorem WithSeminorms.firstCountableTopology (hp : WithSeminorms p) :
    FirstCountableTopology E := by
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    ⊢ FirstCountableTopology E
  -/
  have := hp.topologicalAddGroup
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    this : TopologicalAddGroup E
    ⊢ FirstCountableTopology E
  -/
  let _ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    this : TopologicalAddGroup E
    x✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    ⊢ FirstCountableTopology E
  -/
  have : UniformAddGroup E := comm_topologicalAddGroup_is_uniform
  have : (𝓝 (0 : E)).IsCountablyGenerated := by
    rw [p.withSeminorms_iff_nhds_eq_iInf.mp hp]
    exact Filter.iInf.isCountablyGenerated _
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    this✝¹ : TopologicalAddGroup E
    x✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝ : UniformAddGroup E
    this : (nhds 0).IsCountablyGenerated
    ⊢ FirstCountableTopology E
  -/
  have : (uniformity E).IsCountablyGenerated := UniformAddGroup.uniformity_countably_generated
  /-
    𝕜 : Type u_1
    E : Type u_5
    ι : Type u_8
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module 𝕜 E
    inst✝² : Nonempty ι
    inst✝¹ : Countable ι
    p : SeminormFamily 𝕜 E ι
    inst✝ : TopologicalSpace E
    hp : WithSeminorms p
    this✝² : TopologicalAddGroup E
    x✝ : UniformSpace E := TopologicalAddGroup.toUniformSpace E
    this✝¹ : UniformAddGroup E
    this✝ : (nhds 0).IsCountablyGenerated
    this : (uniformity E).IsCountablyGenerated
    ⊢ FirstCountableTopology E
  -/
  exact UniformSpace.firstCountableTopology E
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-13")] alias
WithSeminorms.first_countable := WithSeminorms.firstCountableTopology


