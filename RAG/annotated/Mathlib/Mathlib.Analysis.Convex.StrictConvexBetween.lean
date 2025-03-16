theorem Sbtw.dist_lt_max_dist (p : P) {p₁ p₂ p₃ : P} (h : Sbtw ℝ p₁ p₂ p₃) :
    dist p₂ p < max (dist p₁ p) (dist p₃ p) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : Sbtw Real p₁ p₂ p₃
    ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  have hp₁p₃ : p₁ -ᵥ p ≠ p₃ -ᵥ p := by simpa using h.left_ne_right
  rw [Sbtw, ← wbtw_vsub_const_iff p, Wbtw, affineSegment_eq_segment, ← insert_endpoints_openSegment,
    Set.mem_insert_iff, Set.mem_insert_iff] at h
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : And (Or (Eq (VSub.vsub p₂ p) (VSub.vsub p₁ p)) (Or (Eq (VSub.vsub p₂ p) (V …
    hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
    ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  rcases h with ⟨h | h | h, hp₂p₁, hp₂p₃⟩
    /-
      case intro.inl.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Eq (VSub.vsub p₂ p) (VSub.vsub p₁ p)
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
  · rw [vsub_left_cancel_iff] at h
    /-
      case intro.inl.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Eq p₂ p₁
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
    exact False.elim (hp₂p₁ h)
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inl.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Eq (VSub.vsub p₂ p) (VSub.vsub p₃ p)
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
  · rw [vsub_left_cancel_iff] at h
    /-
      case intro.inr.inl.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Eq p₂ p₃
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
    exact False.elim (hp₂p₃ h)
    /-
      🎉 no goals
    -/
    /-
      case intro.inr.inr.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Membership.mem (openSegment Real (VSub.vsub p₁ p) (VSub.vsub p₃ p)) (VSub. …
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
  · rw [openSegment_eq_image, Set.mem_image] at h
    /-
      case intro.inr.inr.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      h : Exists fun x => And (Membership.mem (Set.Ioo 0 1) x) (Eq (HAdd.hAdd (HSMul …
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
    rcases h with ⟨r, ⟨hr0, hr1⟩, hr⟩
    /-
      case intro.inr.inr.intro.intro.intro.intro
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      hp₁p₃ : Ne (VSub.vsub p₁ p) (VSub.vsub p₃ p)
      hp₂p₁ : Ne p₂ p₁
      hp₂p₃ : Ne p₂ p₃
      r : Real
      hr : Eq (HAdd.hAdd (HSMul.hSMul (HSub.hSub 1 r) (VSub.vsub p₁ p)) (HSMul.hSMul …
      hr0 : LT.lt 0 r
      hr1 : LT.lt r 1
      ⊢ LT.lt (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
    -/
    simp_rw [@dist_eq_norm_vsub V, ← hr]
    exact
      norm_combo_lt_of_ne (le_max_left _ _) (le_max_right _ _) hp₁p₃ (sub_pos.2 hr1) hr0 (by abel)


theorem Wbtw.dist_le_max_dist (p : P) {p₁ p₂ p₃ : P} (h : Wbtw ℝ p₁ p₂ p₃) :
    dist p₂ p ≤ max (dist p₁ p) (dist p₃ p) := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    ⊢ LE.le (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  by_cases hp₁ : p₂ = p₁; · simp [hp₁]
                            /-
                              🎉 no goals
                            -/
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    hp₁ : Not (Eq p₂ p₁)
    ⊢ LE.le (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  by_cases hp₃ : p₂ = p₃; · simp [hp₃]
                            /-
                              🎉 no goals
                            -/
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    hp₁ : Not (Eq p₂ p₁)
    hp₃ : Not (Eq p₂ p₃)
    ⊢ LE.le (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  have hs : Sbtw ℝ p₁ p₂ p₃ := ⟨h, hp₁, hp₃⟩
  /-
    case neg
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    h : Wbtw Real p₁ p₂ p₃
    hp₁ : Not (Eq p₂ p₁)
    hp₃ : Not (Eq p₂ p₃)
    hs : Sbtw Real p₁ p₂ p₃
    ⊢ LE.le (Dist.dist p₂ p) (Max.max (Dist.dist p₁ p) (Dist.dist p₃ p))
  -/
  exact (hs.dist_lt_max_dist _).le
  /-
    🎉 no goals
  -/


/-- Given three collinear points, two (not equal) with distance `r` from `p` and one with
distance at most `r` from `p`, the third point is weakly between the other two points. -/
theorem Collinear.wbtw_of_dist_eq_of_dist_le {p p₁ p₂ p₃ : P} {r : ℝ}
    (h : Collinear ℝ ({p₁, p₂, p₃} : Set P)) (hp₁ : dist p₁ p = r) (hp₂ : dist p₂ p ≤ r)
    (hp₃ : dist p₃ p = r) (hp₁p₃ : p₁ ≠ p₃) : Wbtw ℝ p₁ p₂ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    r : Real
    h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
    hp₁ : Eq (Dist.dist p₁ p) r
    hp₂ : LE.le (Dist.dist p₂ p) r
    hp₃ : Eq (Dist.dist p₃ p) r
    hp₁p₃ : Ne p₁ p₃
    ⊢ Wbtw Real p₁ p₂ p₃
  -/
  rcases h.wbtw_or_wbtw_or_wbtw with (hw | hw | hw)
    /-
      case inl
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₁ p₂ p₃
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
  · exact hw
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₂ p₃ p₁
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
  · by_cases hp₃p₂ : p₃ = p₂
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝⁴ : NormedAddCommGroup V
        inst✝³ : NormedSpace Real V
        inst✝² : StrictConvexSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        p p₁ p₂ p₃ : P
        r : Real
        h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
        hp₁ : Eq (Dist.dist p₁ p) r
        hp₂ : LE.le (Dist.dist p₂ p) r
        hp₃ : Eq (Dist.dist p₃ p) r
        hp₁p₃ : Ne p₁ p₃
        hw : Wbtw Real p₂ p₃ p₁
        hp₃p₂ : Eq p₃ p₂
        ⊢ Wbtw Real p₁ p₂ p₃
      -/
    · simp [hp₃p₂]
      /-
        🎉 no goals
      -/
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₂ p₃ p₁
      hp₃p₂ : Not (Eq p₃ p₂)
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    have hs : Sbtw ℝ p₂ p₃ p₁ := ⟨hw, hp₃p₂, hp₁p₃.symm⟩
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₂ p₃ p₁
      hp₃p₂ : Not (Eq p₃ p₂)
      hs : Sbtw Real p₂ p₃ p₁
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    have hs' := hs.dist_lt_max_dist p
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₂ p₃ p₁
      hp₃p₂ : Not (Eq p₃ p₂)
      hs : Sbtw Real p₂ p₃ p₁
      hs' : LT.lt (Dist.dist p₃ p) (Max.max (Dist.dist p₂ p) (Dist.dist p₁ p))
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    rw [hp₁, hp₃, lt_max_iff, lt_self_iff_false, or_false] at hs'
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₂ p₃ p₁
      hp₃p₂ : Not (Eq p₃ p₂)
      hs : Sbtw Real p₂ p₃ p₁
      hs' : LT.lt r (Dist.dist p₂ p)
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    exact False.elim (hp₂.not_lt hs')
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₃ p₁ p₂
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
  · by_cases hp₁p₂ : p₁ = p₂
      /-
        case pos
        V : Type u_1
        P : Type u_2
        inst✝⁴ : NormedAddCommGroup V
        inst✝³ : NormedSpace Real V
        inst✝² : StrictConvexSpace Real V
        inst✝¹ : PseudoMetricSpace P
        inst✝ : NormedAddTorsor V P
        p p₁ p₂ p₃ : P
        r : Real
        h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
        hp₁ : Eq (Dist.dist p₁ p) r
        hp₂ : LE.le (Dist.dist p₂ p) r
        hp₃ : Eq (Dist.dist p₃ p) r
        hp₁p₃ : Ne p₁ p₃
        hw : Wbtw Real p₃ p₁ p₂
        hp₁p₂ : Eq p₁ p₂
        ⊢ Wbtw Real p₁ p₂ p₃
      -/
    · simp [hp₁p₂]
      /-
        🎉 no goals
      -/
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₃ p₁ p₂
      hp₁p₂ : Not (Eq p₁ p₂)
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    have hs : Sbtw ℝ p₃ p₁ p₂ := ⟨hw, hp₁p₃, hp₁p₂⟩
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₃ p₁ p₂
      hp₁p₂ : Not (Eq p₁ p₂)
      hs : Sbtw Real p₃ p₁ p₂
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    have hs' := hs.dist_lt_max_dist p
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₃ p₁ p₂
      hp₁p₂ : Not (Eq p₁ p₂)
      hs : Sbtw Real p₃ p₁ p₂
      hs' : LT.lt (Dist.dist p₁ p) (Max.max (Dist.dist p₃ p) (Dist.dist p₂ p))
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    rw [hp₁, hp₃, lt_max_iff, lt_self_iff_false, false_or] at hs'
    /-
      case neg
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LE.le (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      hw : Wbtw Real p₃ p₁ p₂
      hp₁p₂ : Not (Eq p₁ p₂)
      hs : Sbtw Real p₃ p₁ p₂
      hs' : LT.lt r (Dist.dist p₂ p)
      ⊢ Wbtw Real p₁ p₂ p₃
    -/
    exact False.elim (hp₂.not_lt hs')
    /-
      🎉 no goals
    -/


/-- Given three collinear points, two (not equal) with distance `r` from `p` and one with
distance less than `r` from `p`, the third point is strictly between the other two points. -/
theorem Collinear.sbtw_of_dist_eq_of_dist_lt {p p₁ p₂ p₃ : P} {r : ℝ}
    (h : Collinear ℝ ({p₁, p₂, p₃} : Set P)) (hp₁ : dist p₁ p = r) (hp₂ : dist p₂ p < r)
    (hp₃ : dist p₃ p = r) (hp₁p₃ : p₁ ≠ p₃) : Sbtw ℝ p₁ p₂ p₃ := by
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p p₁ p₂ p₃ : P
    r : Real
    h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
    hp₁ : Eq (Dist.dist p₁ p) r
    hp₂ : LT.lt (Dist.dist p₂ p) r
    hp₃ : Eq (Dist.dist p₃ p) r
    hp₁p₃ : Ne p₁ p₃
    ⊢ Sbtw Real p₁ p₂ p₃
  -/
  refine ⟨h.wbtw_of_dist_eq_of_dist_le hp₁ hp₂.le hp₃ hp₁p₃, ?_, ?_⟩
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LT.lt (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      ⊢ Ne p₂ p₁
    -/
  · rintro rfl
    /-
      case refine_1
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₂ p₃ : P
      r : Real
      hp₂ : LT.lt (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      h : Collinear Real (Insert.insert p₂ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₂ p) r
      hp₁p₃ : Ne p₂ p₃
      ⊢ False
    -/
    exact hp₂.ne hp₁
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ p₃ : P
      r : Real
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₃ …
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LT.lt (Dist.dist p₂ p) r
      hp₃ : Eq (Dist.dist p₃ p) r
      hp₁p₃ : Ne p₁ p₃
      ⊢ Ne p₂ p₃
    -/
  · rintro rfl
    /-
      case refine_2
      V : Type u_1
      P : Type u_2
      inst✝⁴ : NormedAddCommGroup V
      inst✝³ : NormedSpace Real V
      inst✝² : StrictConvexSpace Real V
      inst✝¹ : PseudoMetricSpace P
      inst✝ : NormedAddTorsor V P
      p p₁ p₂ : P
      r : Real
      hp₁ : Eq (Dist.dist p₁ p) r
      hp₂ : LT.lt (Dist.dist p₂ p) r
      h : Collinear Real (Insert.insert p₁ (Insert.insert p₂ (Singleton.singleton p₂ …
      hp₃ : Eq (Dist.dist p₂ p) r
      hp₁p₃ : Ne p₁ p₂
      ⊢ False
    -/
    exact hp₂.ne hp₃
    /-
      🎉 no goals
    -/


/-- In a strictly convex space, the triangle inequality turns into an equality if and only if the
middle point belongs to the segment joining two other points. -/
lemma dist_add_dist_eq_iff : dist a b + dist b c = dist a c ↔ Wbtw ℝ a b c := by
  have :
      dist (a -ᵥ a) (b -ᵥ a) + dist (b -ᵥ a) (c -ᵥ a) = dist (a -ᵥ a) (c -ᵥ a) ↔
        b -ᵥ a ∈ segment ℝ (a -ᵥ a) (c -ᵥ a) := by
    simp only [mem_segment_iff_sameRay, sameRay_iff_norm_add, dist_eq_norm', sub_add_sub_cancel',
      eq_comm]
  simp_rw [dist_vsub_cancel_right, ← affineSegment_eq_segment, ← affineSegment_vsub_const_image]
    at this
  /-
    V : Type u_1
    P : Type u_2
    inst✝⁴ : NormedAddCommGroup V
    inst✝³ : NormedSpace Real V
    inst✝² : StrictConvexSpace Real V
    inst✝¹ : MetricSpace P
    inst✝ : NormedAddTorsor V P
    a b c : P
    this : Iff (Eq (HAdd.hAdd (Dist.dist a b) (Dist.dist b c)) (Dist.dist a c)) (M …
    ⊢ Iff (Eq (HAdd.hAdd (Dist.dist a b) (Dist.dist b c)) (Dist.dist a c)) (Wbtw R …
  -/
  rwa [(vsub_left_injective _).mem_set_image] at this
  /-
    🎉 no goals
  -/


lemma eq_lineMap_of_dist_eq_mul_of_dist_eq_mul (hxy : dist x y = r * dist x z)
    (hyz : dist y z = (1 - r) * dist x z) : y = AffineMap.lineMap x z r := by
  have : y -ᵥ x ∈ [(0 : E) -[ℝ] z -ᵥ x] := by
    rw [mem_segment_iff_wbtw, ← dist_add_dist_eq_iff, dist_zero_left, dist_vsub_cancel_right,
      ← dist_eq_norm_vsub', ← dist_eq_norm_vsub', hxy, hyz, ← add_mul, add_sub_cancel,
      one_mul]
  /-
    E : Type u_3
    PE : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : StrictConvexSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    r : Real
    x y z : PE
    hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
    hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
    this : Membership.mem (segment Real 0 (VSub.vsub z x)) (VSub.vsub y x)
    ⊢ Eq y ((AffineMap.lineMap x z) r)
  -/
  obtain rfl | hne := eq_or_ne x z
    /-
      case inl
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x x))
      hyz : Eq (Dist.dist y x) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x x))
      this : Membership.mem (segment Real 0 (VSub.vsub x x)) (VSub.vsub y x)
      ⊢ Eq y ((AffineMap.lineMap x x) r)
    -/
  · obtain rfl : y = x := by simpa
    /-
      case inl
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      y : PE
      hxy : Eq (Dist.dist y y) (HMul.hMul r (Dist.dist y y))
      hyz : Eq (Dist.dist y y) (HMul.hMul (HSub.hSub 1 r) (Dist.dist y y))
      this : Membership.mem (segment Real 0 (VSub.vsub y y)) (VSub.vsub y y)
      ⊢ Eq y ((AffineMap.lineMap y y) r)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y z : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
      hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
      this : Membership.mem (segment Real 0 (VSub.vsub z x)) (VSub.vsub y x)
      hne : Ne x z
      ⊢ Eq y ((AffineMap.lineMap x z) r)
    -/
  · rw [← dist_ne_zero] at hne
    /-
      case inr
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y z : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
      hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
      this : Membership.mem (segment Real 0 (VSub.vsub z x)) (VSub.vsub y x)
      hne : Ne (Dist.dist x z) 0
      ⊢ Eq y ((AffineMap.lineMap x z) r)
    -/
    obtain ⟨a, b, _, hb, _, H⟩ := this
    /-
      case inr.intro.intro.intro.intro.intro
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y z : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
      hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
      hne : Ne (Dist.dist x z) 0
      a b : Real
      left✝¹ : LE.le 0 a
      hb : LE.le 0 b
      left✝ : Eq (HAdd.hAdd a b) 1
      H : Eq (HAdd.hAdd (HSMul.hSMul a 0) (HSMul.hSMul b (VSub.vsub z x))) (VSub.vsu …
      ⊢ Eq y ((AffineMap.lineMap x z) r)
    -/
    rw [smul_zero, zero_add] at H
    /-
      case inr.intro.intro.intro.intro.intro
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y z : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
      hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
      hne : Ne (Dist.dist x z) 0
      a b : Real
      left✝¹ : LE.le 0 a
      hb : LE.le 0 b
      left✝ : Eq (HAdd.hAdd a b) 1
      H : Eq (HSMul.hSMul b (VSub.vsub z x)) (VSub.vsub y x)
      ⊢ Eq y ((AffineMap.lineMap x z) r)
    -/
    have H' := congr_arg norm H
    rw [norm_smul, Real.norm_of_nonneg hb, ← dist_eq_norm_vsub', ← dist_eq_norm_vsub', hxy,
      mul_left_inj' hne] at H'
    /-
      case inr.intro.intro.intro.intro.intro
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      r : Real
      x y z : PE
      hxy : Eq (Dist.dist x y) (HMul.hMul r (Dist.dist x z))
      hyz : Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 r) (Dist.dist x z))
      hne : Ne (Dist.dist x z) 0
      a b : Real
      left✝¹ : LE.le 0 a
      hb : LE.le 0 b
      left✝ : Eq (HAdd.hAdd a b) 1
      H : Eq (HSMul.hSMul b (VSub.vsub z x)) (VSub.vsub y x)
      H' : Eq b r
      ⊢ Eq y ((AffineMap.lineMap x z) r)
    -/
    rw [AffineMap.lineMap_apply, ← H', H, vsub_vadd]
    /-
      🎉 no goals
    -/


lemma eq_midpoint_of_dist_eq_half (hx : dist x y = dist x z / 2) (hy : dist y z = dist x z / 2) :
    y = midpoint ℝ x z := by
  /-
    E : Type u_3
    PE : Type u_5
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace Real E
    inst✝² : StrictConvexSpace Real E
    inst✝¹ : MetricSpace PE
    inst✝ : NormedAddTorsor E PE
    x y z : PE
    hx : Eq (Dist.dist x y) (HDiv.hDiv (Dist.dist x z) 2)
    hy : Eq (Dist.dist y z) (HDiv.hDiv (Dist.dist x z) 2)
    ⊢ Eq y (midpoint Real x z)
  -/
  apply eq_lineMap_of_dist_eq_mul_of_dist_eq_mul
    /-
      case hxy
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      x y z : PE
      hx : Eq (Dist.dist x y) (HDiv.hDiv (Dist.dist x z) 2)
      hy : Eq (Dist.dist y z) (HDiv.hDiv (Dist.dist x z) 2)
      ⊢ Eq (Dist.dist x y) (HMul.hMul (Invertible.invOf 2) (Dist.dist x z))
    -/
  · rwa [invOf_eq_inv, ← div_eq_inv_mul]
    /-
      🎉 no goals
    -/
    /-
      case hyz
      E : Type u_3
      PE : Type u_5
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace Real E
      inst✝² : StrictConvexSpace Real E
      inst✝¹ : MetricSpace PE
      inst✝ : NormedAddTorsor E PE
      x y z : PE
      hx : Eq (Dist.dist x y) (HDiv.hDiv (Dist.dist x z) 2)
      hy : Eq (Dist.dist y z) (HDiv.hDiv (Dist.dist x z) 2)
      ⊢ Eq (Dist.dist y z) (HMul.hMul (HSub.hSub 1 (Invertible.invOf 2)) (Dist.dist  …
    -/
  · rwa [invOf_eq_inv, ← one_div, sub_half, one_div, ← div_eq_inv_mul]
    /-
      🎉 no goals
    -/


/-- An isometry of `NormedAddTorsor`s for real normed spaces, strictly convex in the case of the
codomain, is an affine isometry.  Unlike Mazur-Ulam, this does not require the isometry to be
surjective. -/
noncomputable def affineIsometryOfStrictConvexSpace (hi : Isometry f) : PF →ᵃⁱ[ℝ] PE :=
  { AffineMap.ofMapMidpoint f
      (fun x y => by
        /-
          V : Type u_1
          P : Type u_2
          inst✝¹¹ : NormedAddCommGroup V
          inst✝¹⁰ : NormedSpace Real V
          inst✝⁹ : StrictConvexSpace Real V
          E : Type u_3
          F : Type u_4
          PE : Type u_5
          PF : Type u_6
          inst✝⁸ : NormedAddCommGroup E
          inst✝⁷ : NormedAddCommGroup F
          inst✝⁶ : NormedSpace Real E
          inst✝⁵ : NormedSpace Real F
          inst✝⁴ : StrictConvexSpace Real E
          inst✝³ : MetricSpace PE
          inst✝² : MetricSpace PF
          inst✝¹ : NormedAddTorsor E PE
          inst✝ : NormedAddTorsor F PF
          r : Real
          f : PF → PE
          x✝ y✝ z : PE
          hi : Isometry f
          x y : PF
          ⊢ Eq (f (midpoint Real x y)) (midpoint Real (f x) (f y))
        -/
        apply eq_midpoint_of_dist_eq_half
          /-
            case hx
            V : Type u_1
            P : Type u_2
            inst✝¹¹ : NormedAddCommGroup V
            inst✝¹⁰ : NormedSpace Real V
            inst✝⁹ : StrictConvexSpace Real V
            E : Type u_3
            F : Type u_4
            PE : Type u_5
            PF : Type u_6
            inst✝⁸ : NormedAddCommGroup E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real E
            inst✝⁵ : NormedSpace Real F
            inst✝⁴ : StrictConvexSpace Real E
            inst✝³ : MetricSpace PE
            inst✝² : MetricSpace PF
            inst✝¹ : NormedAddTorsor E PE
            inst✝ : NormedAddTorsor F PF
            r : Real
            f : PF → PE
            x✝ y✝ z : PE
            hi : Isometry f
            x y : PF
            ⊢ Eq (Dist.dist (f x) (f (midpoint Real x y))) (HDiv.hDiv (Dist.dist (f x) (f  …
          -/
        · rw [hi.dist_eq, hi.dist_eq]
          /-
            case hx
            V : Type u_1
            P : Type u_2
            inst✝¹¹ : NormedAddCommGroup V
            inst✝¹⁰ : NormedSpace Real V
            inst✝⁹ : StrictConvexSpace Real V
            E : Type u_3
            F : Type u_4
            PE : Type u_5
            PF : Type u_6
            inst✝⁸ : NormedAddCommGroup E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real E
            inst✝⁵ : NormedSpace Real F
            inst✝⁴ : StrictConvexSpace Real E
            inst✝³ : MetricSpace PE
            inst✝² : MetricSpace PF
            inst✝¹ : NormedAddTorsor E PE
            inst✝ : NormedAddTorsor F PF
            r : Real
            f : PF → PE
            x✝ y✝ z : PE
            hi : Isometry f
            x y : PF
            ⊢ Eq (Dist.dist x (midpoint Real x y)) (HDiv.hDiv (Dist.dist x y) 2)
          -/
          simp only [dist_left_midpoint, Real.norm_of_nonneg zero_le_two, div_eq_inv_mul]
          /-
            🎉 no goals
          -/
          /-
            case hy
            V : Type u_1
            P : Type u_2
            inst✝¹¹ : NormedAddCommGroup V
            inst✝¹⁰ : NormedSpace Real V
            inst✝⁹ : StrictConvexSpace Real V
            E : Type u_3
            F : Type u_4
            PE : Type u_5
            PF : Type u_6
            inst✝⁸ : NormedAddCommGroup E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real E
            inst✝⁵ : NormedSpace Real F
            inst✝⁴ : StrictConvexSpace Real E
            inst✝³ : MetricSpace PE
            inst✝² : MetricSpace PF
            inst✝¹ : NormedAddTorsor E PE
            inst✝ : NormedAddTorsor F PF
            r : Real
            f : PF → PE
            x✝ y✝ z : PE
            hi : Isometry f
            x y : PF
            ⊢ Eq (Dist.dist (f (midpoint Real x y)) (f y)) (HDiv.hDiv (Dist.dist (f x) (f  …
          -/
        · rw [hi.dist_eq, hi.dist_eq]
          /-
            case hy
            V : Type u_1
            P : Type u_2
            inst✝¹¹ : NormedAddCommGroup V
            inst✝¹⁰ : NormedSpace Real V
            inst✝⁹ : StrictConvexSpace Real V
            E : Type u_3
            F : Type u_4
            PE : Type u_5
            PF : Type u_6
            inst✝⁸ : NormedAddCommGroup E
            inst✝⁷ : NormedAddCommGroup F
            inst✝⁶ : NormedSpace Real E
            inst✝⁵ : NormedSpace Real F
            inst✝⁴ : StrictConvexSpace Real E
            inst✝³ : MetricSpace PE
            inst✝² : MetricSpace PF
            inst✝¹ : NormedAddTorsor E PE
            inst✝ : NormedAddTorsor F PF
            r : Real
            f : PF → PE
            x✝ y✝ z : PE
            hi : Isometry f
            x y : PF
            ⊢ Eq (Dist.dist (midpoint Real x y) y) (HDiv.hDiv (Dist.dist x y) 2)
          -/
          simp only [dist_midpoint_right, Real.norm_of_nonneg zero_le_two, div_eq_inv_mul])
          /-
            🎉 no goals
          -/
      hi.continuous with
                            /-
                              V : Type u_1
                              P : Type u_2
                              inst✝¹¹ : NormedAddCommGroup V
                              inst✝¹⁰ : NormedSpace Real V
                              inst✝⁹ : StrictConvexSpace Real V
                              E : Type u_3
                              F : Type u_4
                              PE : Type u_5
                              PF : Type u_6
                              inst✝⁸ : NormedAddCommGroup E
                              inst✝⁷ : NormedAddCommGroup F
                              inst✝⁶ : NormedSpace Real E
                              inst✝⁵ : NormedSpace Real F
                              inst✝⁴ : StrictConvexSpace Real E
                              inst✝³ : MetricSpace PE
                              inst✝² : MetricSpace PF
                              inst✝¹ : NormedAddTorsor E PE
                              inst✝ : NormedAddTorsor F PF
                              r : Real
                              f : PF → PE
                              x✝ y z : PE
                              hi : Isometry f
                              x : F
                              ⊢ Eq (Norm.norm (__src✝.linear x)) (Norm.norm x)
                            -/
    norm_map := fun x => by simp [AffineMap.ofMapMidpoint, ← dist_eq_norm_vsub E, hi.dist_eq] }
                            /-
                              🎉 no goals
                            -/


@[simp] lemma coe_affineIsometryOfStrictConvexSpace (hi : Isometry f) :
    ⇑hi.affineIsometryOfStrictConvexSpace = f := rfl


@[simp] lemma affineIsometryOfStrictConvexSpace_apply (hi : Isometry f) (p : PF) :
    hi.affineIsometryOfStrictConvexSpace p = f p := rfl


