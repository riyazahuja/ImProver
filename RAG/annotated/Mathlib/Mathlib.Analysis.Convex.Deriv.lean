/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is differentiable on its interior,
and `f'` is monotone on the interior, then `f` is convex on `D`. -/
theorem MonotoneOn.convexOn_of_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D))
    (hf'_mono : MonotoneOn (deriv f) (interior D)) : ConvexOn ℝ D f :=
  convexOn_of_slope_mono_adjacent hD
    (by
      /-
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        ⊢ ∀ {x y z : Real}, Membership.mem D x → Membership.mem D z → LT.lt x y → LT.l …
      -/
      intro x y z hx hz hxy hyz
      -- First we prove some trivial inclusions
      /-
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        x y z : Real
        hx : Membership.mem D x
        hz : Membership.mem D z
        hxy : LT.lt x y
        hyz : LT.lt y z
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
      -/
      have hxzD : Icc x z ⊆ D := hD.ordConnected.out hx hz
      /-
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        x y z : Real
        hx : Membership.mem D x
        hz : Membership.mem D z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hxzD : HasSubset.Subset (Set.Icc x z) D
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
      -/
      have hxyD : Icc x y ⊆ D := (Icc_subset_Icc_right hyz.le).trans hxzD
      have hxyD' : Ioo x y ⊆ interior D :=
        subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hxyD⟩
      /-
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        x y z : Real
        hx : Membership.mem D x
        hz : Membership.mem D z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hxzD : HasSubset.Subset (Set.Icc x z) D
        hxyD : HasSubset.Subset (Set.Icc x y) D
        hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
      -/
      have hyzD : Icc y z ⊆ D := (Icc_subset_Icc_left hxy.le).trans hxzD
      have hyzD' : Ioo y z ⊆ interior D :=
        subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hyzD⟩
      -- Then we apply MVT to both `[x, y]` and `[y, z]`
      obtain ⟨a, ⟨hxa, hay⟩, ha⟩ : ∃ a ∈ Ioo x y, deriv f a = (f y - f x) / (y - x) :=
        exists_deriv_eq_slope f hxy (hf.mono hxyD) (hf'.mono hxyD')
      obtain ⟨b, ⟨hyb, hbz⟩, hb⟩ : ∃ b ∈ Ioo y z, deriv f b = (f z - f y) / (z - y) :=
        exists_deriv_eq_slope f hyz (hf.mono hyzD) (hf'.mono hyzD')
      /-
        case intro.intro.intro.intro.intro.intro
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        x y z : Real
        hx : Membership.mem D x
        hz : Membership.mem D z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hxzD : HasSubset.Subset (Set.Icc x z) D
        hxyD : HasSubset.Subset (Set.Icc x y) D
        hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
        hyzD : HasSubset.Subset (Set.Icc y z) D
        hyzD' : HasSubset.Subset (Set.Ioo y z) (interior D)
        a : Real
        ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
        hxa : LT.lt x a
        hay : LT.lt a y
        b : Real
        hb : Eq (deriv f b) (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y))
        hyb : LT.lt y b
        hbz : LT.lt b z
        ⊢ LE.le (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
      -/
      rw [← ha, ← hb]
      /-
        case intro.intro.intro.intro.intro.intro
        D : Set Real
        hD : Convex Real D
        f : Real → Real
        hf : ContinuousOn f D
        hf' : DifferentiableOn Real f (interior D)
        hf'_mono : MonotoneOn (deriv f) (interior D)
        x y z : Real
        hx : Membership.mem D x
        hz : Membership.mem D z
        hxy : LT.lt x y
        hyz : LT.lt y z
        hxzD : HasSubset.Subset (Set.Icc x z) D
        hxyD : HasSubset.Subset (Set.Icc x y) D
        hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
        hyzD : HasSubset.Subset (Set.Icc y z) D
        hyzD' : HasSubset.Subset (Set.Ioo y z) (interior D)
        a : Real
        ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
        hxa : LT.lt x a
        hay : LT.lt a y
        b : Real
        hb : Eq (deriv f b) (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y))
        hyb : LT.lt y b
        hbz : LT.lt b z
        ⊢ LE.le (deriv f a) (deriv f b)
      -/
      exact hf'_mono (hxyD' ⟨hxa, hay⟩) (hyzD' ⟨hyb, hbz⟩) (hay.trans hyb).le)
      /-
        🎉 no goals
      -/


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is differentiable on its interior,
and `f'` is antitone on the interior, then `f` is concave on `D`. -/
theorem AntitoneOn.concaveOn_of_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : DifferentiableOn ℝ f (interior D))
    (h_anti : AntitoneOn (deriv f) (interior D)) : ConcaveOn ℝ D f :=
  haveI : MonotoneOn (deriv (-f)) (interior D) := by
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : DifferentiableOn Real f (interior D)
      h_anti : AntitoneOn (deriv f) (interior D)
      ⊢ MonotoneOn (deriv (Neg.neg f)) (interior D)
    -/
    simpa only [← deriv.neg] using h_anti.neg
    /-
      🎉 no goals
    -/
  neg_convexOn_iff.mp (this.convexOn_of_deriv hD hf.neg hf'.neg)


theorem StrictMonoOn.exists_slope_lt_deriv_aux {x y : ℝ} {f : ℝ → ℝ} (hf : ContinuousOn f (Icc x y))
    (hxy : x < y) (hf'_mono : StrictMonoOn (deriv f) (Ioo x y)) (h : ∀ w ∈ Ioo x y, deriv f w ≠ 0) :
    ∃ a ∈ Ioo x y, (f y - f x) / (y - x) < deriv f a := by
  have A : DifferentiableOn ℝ f (Ioo x y) := fun w wmem =>
    (differentiableAt_of_deriv_ne_zero (h w wmem)).differentiableWithinAt
  obtain ⟨a, ⟨hxa, hay⟩, ha⟩ : ∃ a ∈ Ioo x y, deriv f a = (f y - f x) / (y - x) :=
    exists_deriv_eq_slope f hxy hf A
  /-
    case intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
  -/
  rcases nonempty_Ioo.2 hay with ⟨b, ⟨hab, hby⟩⟩
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hab : LT.lt a b
    hby : LT.lt b y
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
  -/
  refine ⟨b, ⟨hxa.trans hab, hby⟩, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hab : LT.lt a b
    hby : LT.lt b y
    ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (deriv f b)
  -/
  rw [← ha]
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hab : LT.lt a b
    hby : LT.lt b y
    ⊢ LT.lt (deriv f a) (deriv f b)
  -/
  exact hf'_mono ⟨hxa, hay⟩ ⟨hxa.trans hab, hby⟩ hab
  /-
    🎉 no goals
  -/


theorem StrictMonoOn.exists_slope_lt_deriv {x y : ℝ} {f : ℝ → ℝ} (hf : ContinuousOn f (Icc x y))
    (hxy : x < y) (hf'_mono : StrictMonoOn (deriv f) (Ioo x y)) :
    ∃ a ∈ Ioo x y, (f y - f x) / (y - x) < deriv f a := by
  /-
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
  -/
  by_cases h : ∀ w ∈ Ioo x y, deriv f w ≠ 0
    /-
      case pos
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
    -/
  · apply StrictMonoOn.exists_slope_lt_deriv_aux hf hxy hf'_mono h
    /-
      🎉 no goals
    -/
    /-
      case neg
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : Not (∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0)
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
    -/
  · push_neg at h
    /-
      case neg
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : Exists fun w => And (Membership.mem (Set.Ioo x y) w) (Eq (deriv f w) 0)
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
    -/
    rcases h with ⟨w, ⟨hxw, hwy⟩, hw⟩
    obtain ⟨a, ⟨hxa, haw⟩, ha⟩ : ∃ a ∈ Ioo x w, (f w - f x) / (w - x) < deriv f a := by
      apply StrictMonoOn.exists_slope_lt_deriv_aux _ hxw _ _
      · exact hf.mono (Icc_subset_Icc le_rfl hwy.le)
      · exact hf'_mono.mono (Ioo_subset_Ioo le_rfl hwy.le)
      · intro z hz
        rw [← hw]
        apply ne_of_lt
        exact hf'_mono ⟨hz.1, hz.2.trans hwy⟩ ⟨hxw, hwy⟩ hz.2
    obtain ⟨b, ⟨hwb, hby⟩, hb⟩ : ∃ b ∈ Ioo w y, (f y - f w) / (y - w) < deriv f b := by
      apply StrictMonoOn.exists_slope_lt_deriv_aux _ hwy _ _
      · refine hf.mono (Icc_subset_Icc hxw.le le_rfl)
      · exact hf'_mono.mono (Ioo_subset_Ioo hxw.le le_rfl)
      · intro z hz
        rw [← hw]
        apply ne_of_gt
        exact hf'_mono ⟨hxw, hwy⟩ ⟨hxw.trans hz.1, hz.2⟩ hz.1
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      ha : LT.lt (HDiv.hDiv (HSub.hSub (f w) (f x)) (HSub.hSub w x)) (deriv f a)
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hb : LT.lt (HDiv.hDiv (HSub.hSub (f y) (f w)) (HSub.hSub y w)) (deriv f b)
      hwb : LT.lt w b
      hby : LT.lt b y
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (HDiv.hDiv (HSub …
    -/
    refine ⟨b, ⟨hxw.trans hwb, hby⟩, ?_⟩
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      ha : LT.lt (HDiv.hDiv (HSub.hSub (f w) (f x)) (HSub.hSub w x)) (deriv f a)
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hb : LT.lt (HDiv.hDiv (HSub.hSub (f y) (f w)) (HSub.hSub y w)) (deriv f b)
      hwb : LT.lt w b
      hby : LT.lt b y
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (deriv f b)
    -/
    simp only [div_lt_iff₀, hxy, hxw, hwy, sub_pos] at ha hb ⊢
    have : deriv f a * (w - x) < deriv f b * (w - x) := by
      apply mul_lt_mul _ le_rfl (sub_pos.2 hxw) _
      · exact hf'_mono ⟨hxa, haw.trans hwy⟩ ⟨hxw.trans hwb, hby⟩ (haw.trans hwb)
      · rw [← hw]
        exact (hf'_mono ⟨hxw, hwy⟩ ⟨hxw.trans hwb, hby⟩ hwb).le
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hwb : LT.lt w b
      hby : LT.lt b y
      ha : LT.lt (HSub.hSub (f w) (f x)) (HMul.hMul (deriv f a) (HSub.hSub w x))
      hb : LT.lt (HSub.hSub (f y) (f w)) (HMul.hMul (deriv f b) (HSub.hSub y w))
      this : LT.lt (HMul.hMul (deriv f a) (HSub.hSub w x)) (HMul.hMul (deriv f b) (H …
      ⊢ LT.lt (HSub.hSub (f y) (f x)) (HMul.hMul (deriv f b) (HSub.hSub y x))
    -/
    linarith
    /-
      🎉 no goals
    -/


theorem StrictMonoOn.exists_deriv_lt_slope_aux {x y : ℝ} {f : ℝ → ℝ} (hf : ContinuousOn f (Icc x y))
    (hxy : x < y) (hf'_mono : StrictMonoOn (deriv f) (Ioo x y)) (h : ∀ w ∈ Ioo x y, deriv f w ≠ 0) :
    ∃ a ∈ Ioo x y, deriv f a < (f y - f x) / (y - x) := by
  have A : DifferentiableOn ℝ f (Ioo x y) := fun w wmem =>
    (differentiableAt_of_deriv_ne_zero (h w wmem)).differentiableWithinAt
  obtain ⟨a, ⟨hxa, hay⟩, ha⟩ : ∃ a ∈ Ioo x y, deriv f a = (f y - f x) / (y - x) :=
    exists_deriv_eq_slope f hxy hf A
  /-
    case intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
  -/
  rcases nonempty_Ioo.2 hxa with ⟨b, ⟨hxb, hba⟩⟩
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hxb : LT.lt x b
    hba : LT.lt b a
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
  -/
  refine ⟨b, ⟨hxb, hba.trans hay⟩, ?_⟩
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hxb : LT.lt x b
    hba : LT.lt b a
    ⊢ LT.lt (deriv f b) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
  -/
  rw [← ha]
  /-
    case intro.intro.intro.intro.intro
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
    A : DifferentiableOn Real f (Set.Ioo x y)
    a : Real
    ha : Eq (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    hxa : LT.lt x a
    hay : LT.lt a y
    b : Real
    hxb : LT.lt x b
    hba : LT.lt b a
    ⊢ LT.lt (deriv f b) (deriv f a)
  -/
  exact hf'_mono ⟨hxb, hba.trans hay⟩ ⟨hxa, hay⟩ hba
  /-
    🎉 no goals
  -/


theorem StrictMonoOn.exists_deriv_lt_slope {x y : ℝ} {f : ℝ → ℝ} (hf : ContinuousOn f (Icc x y))
    (hxy : x < y) (hf'_mono : StrictMonoOn (deriv f) (Ioo x y)) :
    ∃ a ∈ Ioo x y, deriv f a < (f y - f x) / (y - x) := by
  /-
    x y : Real
    f : Real → Real
    hf : ContinuousOn f (Set.Icc x y)
    hxy : LT.lt x y
    hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
    ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
  -/
  by_cases h : ∀ w ∈ Ioo x y, deriv f w ≠ 0
    /-
      case pos
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : ∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
    -/
  · apply StrictMonoOn.exists_deriv_lt_slope_aux hf hxy hf'_mono h
    /-
      🎉 no goals
    -/
    /-
      case neg
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : Not (∀ (w : Real), Membership.mem (Set.Ioo x y) w → Ne (deriv f w) 0)
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
    -/
  · push_neg at h
    /-
      case neg
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      h : Exists fun w => And (Membership.mem (Set.Ioo x y) w) (Eq (deriv f w) 0)
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
    -/
    rcases h with ⟨w, ⟨hxw, hwy⟩, hw⟩
    obtain ⟨a, ⟨hxa, haw⟩, ha⟩ : ∃ a ∈ Ioo x w, deriv f a < (f w - f x) / (w - x) := by
      apply StrictMonoOn.exists_deriv_lt_slope_aux _ hxw _ _
      · exact hf.mono (Icc_subset_Icc le_rfl hwy.le)
      · exact hf'_mono.mono (Ioo_subset_Ioo le_rfl hwy.le)
      · intro z hz
        rw [← hw]
        apply ne_of_lt
        exact hf'_mono ⟨hz.1, hz.2.trans hwy⟩ ⟨hxw, hwy⟩ hz.2
    obtain ⟨b, ⟨hwb, hby⟩, hb⟩ : ∃ b ∈ Ioo w y, deriv f b < (f y - f w) / (y - w) := by
      apply StrictMonoOn.exists_deriv_lt_slope_aux _ hwy _ _
      · refine hf.mono (Icc_subset_Icc hxw.le le_rfl)
      · exact hf'_mono.mono (Ioo_subset_Ioo hxw.le le_rfl)
      · intro z hz
        rw [← hw]
        apply ne_of_gt
        exact hf'_mono ⟨hxw, hwy⟩ ⟨hxw.trans hz.1, hz.2⟩ hz.1
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      ha : LT.lt (deriv f a) (HDiv.hDiv (HSub.hSub (f w) (f x)) (HSub.hSub w x))
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hb : LT.lt (deriv f b) (HDiv.hDiv (HSub.hSub (f y) (f w)) (HSub.hSub y w))
      hwb : LT.lt w b
      hby : LT.lt b y
      ⊢ Exists fun a => And (Membership.mem (Set.Ioo x y) a) (LT.lt (deriv f a) (HDi …
    -/
    refine ⟨a, ⟨hxa, haw.trans hwy⟩, ?_⟩
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      ha : LT.lt (deriv f a) (HDiv.hDiv (HSub.hSub (f w) (f x)) (HSub.hSub w x))
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hb : LT.lt (deriv f b) (HDiv.hDiv (HSub.hSub (f y) (f w)) (HSub.hSub y w))
      hwb : LT.lt w b
      hby : LT.lt b y
      ⊢ LT.lt (deriv f a) (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x))
    -/
    simp only [lt_div_iff₀, hxy, hxw, hwy, sub_pos] at ha hb ⊢
    have : deriv f a * (y - w) < deriv f b * (y - w) := by
      apply mul_lt_mul _ le_rfl (sub_pos.2 hwy) _
      · exact hf'_mono ⟨hxa, haw.trans hwy⟩ ⟨hxw.trans hwb, hby⟩ (haw.trans hwb)
      · rw [← hw]
        exact (hf'_mono ⟨hxw, hwy⟩ ⟨hxw.trans hwb, hby⟩ hwb).le
    /-
      case neg.intro.intro.intro.intro.intro.intro.intro.intro.intro
      x y : Real
      f : Real → Real
      hf : ContinuousOn f (Set.Icc x y)
      hxy : LT.lt x y
      hf'_mono : StrictMonoOn (deriv f) (Set.Ioo x y)
      w : Real
      hw : Eq (deriv f w) 0
      hxw : LT.lt x w
      hwy : LT.lt w y
      a : Real
      hxa : LT.lt x a
      haw : LT.lt a w
      b : Real
      hwb : LT.lt w b
      hby : LT.lt b y
      ha : LT.lt (HMul.hMul (deriv f a) (HSub.hSub w x)) (HSub.hSub (f w) (f x))
      hb : LT.lt (HMul.hMul (deriv f b) (HSub.hSub y w)) (HSub.hSub (f y) (f w))
      this : LT.lt (HMul.hMul (deriv f a) (HSub.hSub y w)) (HMul.hMul (deriv f b) (H …
      ⊢ LT.lt (HMul.hMul (deriv f a) (HSub.hSub y x)) (HSub.hSub (f y) (f x))
    -/
    linarith
    /-
      🎉 no goals
    -/


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, and `f'` is strictly monotone on the
interior, then `f` is strictly convex on `D`.
Note that we don't require differentiability, since it is guaranteed at all but at most
one point by the strict monotonicity of `f'`. -/
theorem StrictMonoOn.strictConvexOn_of_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : StrictMonoOn (deriv f) (interior D)) : StrictConvexOn ℝ D f :=
  strictConvexOn_of_slope_strict_mono_adjacent hD fun {x y z} hx hz hxy hyz => by
    -- First we prove some trivial inclusions
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : StrictMonoOn (deriv f) (interior D)
      x y z : Real
      hx : Membership.mem D x
      hz : Membership.mem D z
      hxy : LT.lt x y
      hyz : LT.lt y z
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
    -/
    have hxzD : Icc x z ⊆ D := hD.ordConnected.out hx hz
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : StrictMonoOn (deriv f) (interior D)
      x y z : Real
      hx : Membership.mem D x
      hz : Membership.mem D z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxzD : HasSubset.Subset (Set.Icc x z) D
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
    -/
    have hxyD : Icc x y ⊆ D := (Icc_subset_Icc_right hyz.le).trans hxzD
    have hxyD' : Ioo x y ⊆ interior D :=
      subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hxyD⟩
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : StrictMonoOn (deriv f) (interior D)
      x y z : Real
      hx : Membership.mem D x
      hz : Membership.mem D z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxzD : HasSubset.Subset (Set.Icc x z) D
      hxyD : HasSubset.Subset (Set.Icc x y) D
      hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
    -/
    have hyzD : Icc y z ⊆ D := (Icc_subset_Icc_left hxy.le).trans hxzD
    have hyzD' : Ioo y z ⊆ interior D :=
      subset_sUnion_of_mem ⟨isOpen_Ioo, Ioo_subset_Icc_self.trans hyzD⟩
    -- Then we get points `a` and `b` in each interval `[x, y]` and `[y, z]` where the derivatives
    -- can be compared to the slopes between `x, y` and `y, z` respectively.
    obtain ⟨a, ⟨hxa, hay⟩, ha⟩ : ∃ a ∈ Ioo x y, (f y - f x) / (y - x) < deriv f a :=
      StrictMonoOn.exists_slope_lt_deriv (hf.mono hxyD) hxy (hf'.mono hxyD')
    obtain ⟨b, ⟨hyb, hbz⟩, hb⟩ : ∃ b ∈ Ioo y z, deriv f b < (f z - f y) / (z - y) :=
      StrictMonoOn.exists_deriv_lt_slope (hf.mono hyzD) hyz (hf'.mono hyzD')
    /-
      case intro.intro.intro.intro.intro.intro
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : StrictMonoOn (deriv f) (interior D)
      x y z : Real
      hx : Membership.mem D x
      hz : Membership.mem D z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxzD : HasSubset.Subset (Set.Icc x z) D
      hxyD : HasSubset.Subset (Set.Icc x y) D
      hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
      hyzD : HasSubset.Subset (Set.Icc y z) D
      hyzD' : HasSubset.Subset (Set.Ioo y z) (interior D)
      a : Real
      ha : LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (deriv f a)
      hxa : LT.lt x a
      hay : LT.lt a y
      b : Real
      hb : LT.lt (deriv f b) (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y))
      hyb : LT.lt y b
      hbz : LT.lt b z
      ⊢ LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (HDiv.hDiv (HSub.h …
    -/
    apply ha.trans (lt_trans _ hb)
    /-
      D : Set Real
      hD : Convex Real D
      f : Real → Real
      hf : ContinuousOn f D
      hf' : StrictMonoOn (deriv f) (interior D)
      x y z : Real
      hx : Membership.mem D x
      hz : Membership.mem D z
      hxy : LT.lt x y
      hyz : LT.lt y z
      hxzD : HasSubset.Subset (Set.Icc x z) D
      hxyD : HasSubset.Subset (Set.Icc x y) D
      hxyD' : HasSubset.Subset (Set.Ioo x y) (interior D)
      hyzD : HasSubset.Subset (Set.Icc y z) D
      hyzD' : HasSubset.Subset (Set.Ioo y z) (interior D)
      a : Real
      ha : LT.lt (HDiv.hDiv (HSub.hSub (f y) (f x)) (HSub.hSub y x)) (deriv f a)
      hxa : LT.lt x a
      hay : LT.lt a y
      b : Real
      hb : LT.lt (deriv f b) (HDiv.hDiv (HSub.hSub (f z) (f y)) (HSub.hSub z y))
      hyb : LT.lt y b
      hbz : LT.lt b z
      ⊢ LT.lt (deriv f a) (deriv f b)
    -/
    exact hf' (hxyD' ⟨hxa, hay⟩) (hyzD' ⟨hyb, hbz⟩) (hay.trans hyb)
    /-
      🎉 no goals
    -/


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ` and `f'` is strictly antitone on the
interior, then `f` is strictly concave on `D`.
Note that we don't require differentiability, since it is guaranteed at all but at most
one point by the strict antitonicity of `f'`. -/
theorem StrictAntiOn.strictConcaveOn_of_deriv {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (h_anti : StrictAntiOn (deriv f) (interior D)) :
    StrictConcaveOn ℝ D f :=
                                                      /-
                                                        D : Set Real
                                                        hD : Convex Real D
                                                        f : Real → Real
                                                        hf : ContinuousOn f D
                                                        h_anti : StrictAntiOn (deriv f) (interior D)
                                                        ⊢ StrictMonoOn (deriv (Neg.neg f)) (interior D)
                                                      -/
  have : StrictMonoOn (deriv (-f)) (interior D) := by simpa only [← deriv.neg] using h_anti.neg
                                                      /-
                                                        🎉 no goals
                                                      -/
  neg_neg f ▸ (this.strictConvexOn_of_deriv hD hf.neg).neg


/-- If a function `f` is differentiable and `f'` is monotone on `ℝ` then `f` is convex. -/
theorem Monotone.convexOn_univ_of_deriv {f : ℝ → ℝ} (hf : Differentiable ℝ f)
    (hf'_mono : Monotone (deriv f)) : ConvexOn ℝ univ f :=
  (hf'_mono.monotoneOn _).convexOn_of_deriv convex_univ hf.continuous.continuousOn
    hf.differentiableOn


/-- If a function `f` is differentiable and `f'` is antitone on `ℝ` then `f` is concave. -/
theorem Antitone.concaveOn_univ_of_deriv {f : ℝ → ℝ} (hf : Differentiable ℝ f)
    (hf'_anti : Antitone (deriv f)) : ConcaveOn ℝ univ f :=
  (hf'_anti.antitoneOn _).concaveOn_of_deriv convex_univ hf.continuous.continuousOn
    hf.differentiableOn


/-- If a function `f` is continuous and `f'` is strictly monotone on `ℝ` then `f` is strictly
convex. Note that we don't require differentiability, since it is guaranteed at all but at most
one point by the strict monotonicity of `f'`. -/
theorem StrictMono.strictConvexOn_univ_of_deriv {f : ℝ → ℝ} (hf : Continuous f)
    (hf'_mono : StrictMono (deriv f)) : StrictConvexOn ℝ univ f :=
  (hf'_mono.strictMonoOn _).strictConvexOn_of_deriv convex_univ hf.continuousOn


/-- If a function `f` is continuous and `f'` is strictly antitone on `ℝ` then `f` is strictly
concave. Note that we don't require differentiability, since it is guaranteed at all but at most
one point by the strict antitonicity of `f'`. -/
theorem StrictAnti.strictConcaveOn_univ_of_deriv {f : ℝ → ℝ} (hf : Continuous f)
    (hf'_anti : StrictAnti (deriv f)) : StrictConcaveOn ℝ univ f :=
  (hf'_anti.strictAntiOn _).strictConcaveOn_of_deriv convex_univ hf.continuousOn


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is twice differentiable on its
interior, and `f''` is nonnegative on the interior, then `f` is convex on `D`. -/
theorem convexOn_of_deriv2_nonneg {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ} (hf : ContinuousOn f D)
    (hf' : DifferentiableOn ℝ f (interior D)) (hf'' : DifferentiableOn ℝ (deriv f) (interior D))
    (hf''_nonneg : ∀ x ∈ interior D, 0 ≤ deriv^[2] f x) : ConvexOn ℝ D f :=
                                                                /-
                                                                  D : Set Real
                                                                  hD : Convex Real D
                                                                  f : Real → Real
                                                                  hf : ContinuousOn f D
                                                                  hf' : DifferentiableOn Real f (interior D)
                                                                  hf'' : DifferentiableOn Real (deriv f) (interior D)
                                                                  hf''_nonneg : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (Nat.itera …
                                                                  ⊢ DifferentiableOn Real (deriv f) (interior (interior D))
                                                                -/
  (monotoneOn_of_deriv_nonneg hD.interior hf''.continuousOn (by rwa [interior_interior]) <| by
                                                                /-
                                                                  🎉 no goals
                                                                -/
        /-
          D : Set Real
          hD : Convex Real D
          f : Real → Real
          hf : ContinuousOn f D
          hf' : DifferentiableOn Real f (interior D)
          hf'' : DifferentiableOn Real (deriv f) (interior D)
          hf''_nonneg : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (Nat.itera …
          ⊢ ∀ (x : Real), Membership.mem (interior (interior D)) x → LE.le 0 (deriv (der …
        -/
        rwa [interior_interior]).convexOn_of_deriv
        /-
          🎉 no goals
        -/
    hD hf hf'


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is twice differentiable on its
interior, and `f''` is nonpositive on the interior, then `f` is concave on `D`. -/
theorem concaveOn_of_deriv2_nonpos {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ} (hf : ContinuousOn f D)
    (hf' : DifferentiableOn ℝ f (interior D)) (hf'' : DifferentiableOn ℝ (deriv f) (interior D))
    (hf''_nonpos : ∀ x ∈ interior D, deriv^[2] f x ≤ 0) : ConcaveOn ℝ D f :=
                                                                /-
                                                                  D : Set Real
                                                                  hD : Convex Real D
                                                                  f : Real → Real
                                                                  hf : ContinuousOn f D
                                                                  hf' : DifferentiableOn Real f (interior D)
                                                                  hf'' : DifferentiableOn Real (deriv f) (interior D)
                                                                  hf''_nonpos : ∀ (x : Real), Membership.mem (interior D) x → LE.le (Nat.iterate …
                                                                  ⊢ DifferentiableOn Real (deriv f) (interior (interior D))
                                                                -/
  (antitoneOn_of_deriv_nonpos hD.interior hf''.continuousOn (by rwa [interior_interior]) <| by
                                                                /-
                                                                  🎉 no goals
                                                                -/
        /-
          D : Set Real
          hD : Convex Real D
          f : Real → Real
          hf : ContinuousOn f D
          hf' : DifferentiableOn Real f (interior D)
          hf'' : DifferentiableOn Real (deriv f) (interior D)
          hf''_nonpos : ∀ (x : Real), Membership.mem (interior D) x → LE.le (Nat.iterate …
          ⊢ ∀ (x : Real), Membership.mem (interior (interior D)) x → LE.le (deriv (deriv …
        -/
        rwa [interior_interior]).concaveOn_of_deriv
        /-
          🎉 no goals
        -/
    hD hf hf'


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is twice differentiable on its
interior, and `f''` is nonnegative on the interior, then `f` is convex on `D`. -/
lemma convexOn_of_hasDerivWithinAt2_nonneg {D : Set ℝ} (hD : Convex ℝ D) {f f' f'' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'' : ∀ x ∈ interior D, HasDerivWithinAt f' (f'' x) (interior D) x)
    (hf''₀ : ∀ x ∈ interior D, 0 ≤ f'' x) : ConvexOn ℝ D f := by
  /-
    D : Set Real
    hD : Convex Real D
    f f' f'' : Real → Real
    hf : ContinuousOn f D
    hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
    hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
    hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
    ⊢ ConvexOn Real D f
  -/
  have : (interior D).EqOn (deriv f) f' := deriv_eqOn isOpen_interior hf'
  /-
    D : Set Real
    hD : Convex Real D
    f f' f'' : Real → Real
    hf : ContinuousOn f D
    hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
    hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
    hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
    this : Set.EqOn (deriv f) f' (interior D)
    ⊢ ConvexOn Real D f
  -/
  refine convexOn_of_deriv2_nonneg hD hf (fun x hx ↦ (hf' _ hx).differentiableWithinAt) ?_ ?_
    /-
      case refine_1
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ DifferentiableOn Real (deriv f) (interior D)
    -/
  · rw [differentiableOn_congr this]
    /-
      case refine_1
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ DifferentiableOn Real f' (interior D)
    -/
    exact fun x hx ↦ (hf'' _ hx).differentiableWithinAt
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (Nat.iterate deriv 2 f …
    -/
  · rintro x hx
    /-
      case refine_2
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le 0 (Nat.iterate deriv 2 f x)
    -/
    convert hf''₀ _ hx using 1
    /-
      case h.e'_4
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ Eq (Nat.iterate deriv 2 f x) (f'' x)
    -/
    dsimp
    /-
      case h.e'_4
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ Eq (deriv (deriv f) x) (f'' x)
    -/
    rw [deriv_eqOn isOpen_interior (fun y hy ↦ ?_) hx]
    /-
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le 0 (f'' x)
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      y : Real
      hy : Membership.mem (interior D) y
      ⊢ HasDerivWithinAt (deriv f) (f'' y) (interior D) y
    -/
    exact (hf'' _ hy).congr this <| by rw [this hy]
    /-
      🎉 no goals
    -/


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ`, is twice differentiable on its
interior, and `f''` is nonpositive on the interior, then `f` is concave on `D`. -/
lemma concaveOn_of_hasDerivWithinAt2_nonpos {D : Set ℝ} (hD : Convex ℝ D) {f f' f'' : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf' : ∀ x ∈ interior D, HasDerivWithinAt f (f' x) (interior D) x)
    (hf'' : ∀ x ∈ interior D, HasDerivWithinAt f' (f'' x) (interior D) x)
    (hf''₀ : ∀ x ∈ interior D, f'' x ≤ 0) : ConcaveOn ℝ D f := by
  /-
    D : Set Real
    hD : Convex Real D
    f f' f'' : Real → Real
    hf : ContinuousOn f D
    hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
    hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
    hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
    ⊢ ConcaveOn Real D f
  -/
  have : (interior D).EqOn (deriv f) f' := deriv_eqOn isOpen_interior hf'
  /-
    D : Set Real
    hD : Convex Real D
    f f' f'' : Real → Real
    hf : ContinuousOn f D
    hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
    hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
    hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
    this : Set.EqOn (deriv f) f' (interior D)
    ⊢ ConcaveOn Real D f
  -/
  refine concaveOn_of_deriv2_nonpos hD hf (fun x hx ↦ (hf' _ hx).differentiableWithinAt) ?_ ?_
    /-
      case refine_1
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ DifferentiableOn Real (deriv f) (interior D)
    -/
  · rw [differentiableOn_congr this]
    /-
      case refine_1
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ DifferentiableOn Real f' (interior D)
    -/
    exact fun x hx ↦ (hf'' _ hx).differentiableWithinAt
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      ⊢ ∀ (x : Real), Membership.mem (interior D) x → LE.le (Nat.iterate deriv 2 f x …
    -/
  · rintro x hx
    /-
      case refine_2
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ LE.le (Nat.iterate deriv 2 f x) 0
    -/
    convert hf''₀ _ hx using 1
    /-
      case h.e'_3
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ Eq (Nat.iterate deriv 2 f x) (f'' x)
    -/
    dsimp
    /-
      case h.e'_3
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      ⊢ Eq (deriv (deriv f) x) (f'' x)
    -/
    rw [deriv_eqOn isOpen_interior (fun y hy ↦ ?_) hx]
    /-
      D : Set Real
      hD : Convex Real D
      f f' f'' : Real → Real
      hf : ContinuousOn f D
      hf' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f (f' x)  …
      hf'' : ∀ (x : Real), Membership.mem (interior D) x → HasDerivWithinAt f' (f''  …
      hf''₀ : ∀ (x : Real), Membership.mem (interior D) x → LE.le (f'' x) 0
      this : Set.EqOn (deriv f) f' (interior D)
      x : Real
      hx : Membership.mem (interior D) x
      y : Real
      hy : Membership.mem (interior D) y
      ⊢ HasDerivWithinAt (deriv f) (f'' y) (interior D) y
    -/
    exact (hf'' _ hy).congr this <| by rw [this hy]
    /-
      🎉 no goals
    -/


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ` and `f''` is strictly positive on the
interior, then `f` is strictly convex on `D`.
Note that we don't require twice differentiability explicitly as it is already implied by the second
derivative being strictly positive, except at at most one point. -/
theorem strictConvexOn_of_deriv2_pos {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf'' : ∀ x ∈ interior D, 0 < (deriv^[2] f) x) :
    StrictConvexOn ℝ D f :=
  ((strictMonoOn_of_deriv_pos hD.interior fun z hz =>
          (differentiableAt_of_deriv_ne_zero
                (hf'' z hz).ne').differentiableWithinAt.continuousWithinAt) <|
           /-
             D : Set Real
             hD : Convex Real D
             f : Real → Real
             hf : ContinuousOn f D
             hf'' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt 0 (Nat.iterate deri …
             ⊢ ∀ (x : Real), Membership.mem (interior (interior D)) x → LT.lt 0 (deriv (der …
           -/
        by rwa [interior_interior]).strictConvexOn_of_deriv
           /-
             🎉 no goals
           -/
    hD hf


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ` and `f''` is strictly negative on the
interior, then `f` is strictly concave on `D`.
Note that we don't require twice differentiability explicitly as it already implied by the second
derivative being strictly negative, except at at most one point. -/
theorem strictConcaveOn_of_deriv2_neg {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf'' : ∀ x ∈ interior D, deriv^[2] f x < 0) :
    StrictConcaveOn ℝ D f :=
  ((strictAntiOn_of_deriv_neg hD.interior fun z hz =>
          (differentiableAt_of_deriv_ne_zero
                (hf'' z hz).ne).differentiableWithinAt.continuousWithinAt) <|
           /-
             D : Set Real
             hD : Convex Real D
             f : Real → Real
             hf : ContinuousOn f D
             hf'' : ∀ (x : Real), Membership.mem (interior D) x → LT.lt (Nat.iterate deriv  …
             ⊢ ∀ (x : Real), Membership.mem (interior (interior D)) x → LT.lt (deriv (deriv …
           -/
        by rwa [interior_interior]).strictConcaveOn_of_deriv
           /-
             🎉 no goals
           -/
    hD hf


/-- If a function `f` is twice differentiable on an open convex set `D ⊆ ℝ` and
`f''` is nonnegative on `D`, then `f` is convex on `D`. -/
theorem convexOn_of_deriv2_nonneg' {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf' : DifferentiableOn ℝ f D) (hf'' : DifferentiableOn ℝ (deriv f) D)
    (hf''_nonneg : ∀ x ∈ D, 0 ≤ (deriv^[2] f) x) : ConvexOn ℝ D f :=
  convexOn_of_deriv2_nonneg hD hf'.continuousOn (hf'.mono interior_subset)
    (hf''.mono interior_subset) fun x hx => hf''_nonneg x (interior_subset hx)


/-- If a function `f` is twice differentiable on an open convex set `D ⊆ ℝ` and
`f''` is nonpositive on `D`, then `f` is concave on `D`. -/
theorem concaveOn_of_deriv2_nonpos' {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf' : DifferentiableOn ℝ f D) (hf'' : DifferentiableOn ℝ (deriv f) D)
    (hf''_nonpos : ∀ x ∈ D, deriv^[2] f x ≤ 0) : ConcaveOn ℝ D f :=
  concaveOn_of_deriv2_nonpos hD hf'.continuousOn (hf'.mono interior_subset)
    (hf''.mono interior_subset) fun x hx => hf''_nonpos x (interior_subset hx)


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ` and `f''` is strictly positive on `D`,
then `f` is strictly convex on `D`.
Note that we don't require twice differentiability explicitly as it is already implied by the second
derivative being strictly positive, except at at most one point. -/
theorem strictConvexOn_of_deriv2_pos' {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf'' : ∀ x ∈ D, 0 < (deriv^[2] f) x) : StrictConvexOn ℝ D f :=
  strictConvexOn_of_deriv2_pos hD hf fun x hx => hf'' x (interior_subset hx)


/-- If a function `f` is continuous on a convex set `D ⊆ ℝ` and `f''` is strictly negative on `D`,
then `f` is strictly concave on `D`.
Note that we don't require twice differentiability explicitly as it is already implied by the second
derivative being strictly negative, except at at most one point. -/
theorem strictConcaveOn_of_deriv2_neg' {D : Set ℝ} (hD : Convex ℝ D) {f : ℝ → ℝ}
    (hf : ContinuousOn f D) (hf'' : ∀ x ∈ D, deriv^[2] f x < 0) : StrictConcaveOn ℝ D f :=
  strictConcaveOn_of_deriv2_neg hD hf fun x hx => hf'' x (interior_subset hx)


/-- If a function `f` is twice differentiable on `ℝ`, and `f''` is nonnegative on `ℝ`,
then `f` is convex on `ℝ`. -/
theorem convexOn_univ_of_deriv2_nonneg {f : ℝ → ℝ} (hf' : Differentiable ℝ f)
    (hf'' : Differentiable ℝ (deriv f)) (hf''_nonneg : ∀ x, 0 ≤ (deriv^[2] f) x) :
    ConvexOn ℝ univ f :=
  convexOn_of_deriv2_nonneg' convex_univ hf'.differentiableOn hf''.differentiableOn fun x _ =>
    hf''_nonneg x


/-- If a function `f` is twice differentiable on `ℝ`, and `f''` is nonpositive on `ℝ`,
then `f` is concave on `ℝ`. -/
theorem concaveOn_univ_of_deriv2_nonpos {f : ℝ → ℝ} (hf' : Differentiable ℝ f)
    (hf'' : Differentiable ℝ (deriv f)) (hf''_nonpos : ∀ x, deriv^[2] f x ≤ 0) :
    ConcaveOn ℝ univ f :=
  concaveOn_of_deriv2_nonpos' convex_univ hf'.differentiableOn hf''.differentiableOn fun x _ =>
    hf''_nonpos x


/-- If a function `f` is continuous on `ℝ`, and `f''` is strictly positive on `ℝ`,
then `f` is strictly convex on `ℝ`.
Note that we don't require twice differentiability explicitly as it is already implied by the second
derivative being strictly positive, except at at most one point. -/
theorem strictConvexOn_univ_of_deriv2_pos {f : ℝ → ℝ} (hf : Continuous f)
    (hf'' : ∀ x, 0 < (deriv^[2] f) x) : StrictConvexOn ℝ univ f :=
  strictConvexOn_of_deriv2_pos' convex_univ hf.continuousOn fun x _ => hf'' x


/-- If a function `f` is continuous on `ℝ`, and `f''` is strictly negative on `ℝ`,
then `f` is strictly concave on `ℝ`.
Note that we don't require twice differentiability explicitly as it is already implied by the second
derivative being strictly negative, except at at most one point. -/
theorem strictConcaveOn_univ_of_deriv2_neg {f : ℝ → ℝ} (hf : Continuous f)
    (hf'' : ∀ x, deriv^[2] f x < 0) : StrictConcaveOn ℝ univ f :=
  strictConcaveOn_of_deriv2_neg' convex_univ hf.continuousOn fun x _ => hf'' x


/-- If `f : 𝕜 → 𝕜` is convex on `s`, then for any point `x ∈ s` the slope of the secant line of `f`
through `x` is monotone on `s \ {x}`. -/
lemma ConvexOn.slope_mono (hfc : ConvexOn 𝕜 s f) (hx : x ∈ s) : MonotoneOn (slope f x) (s \ {x}) :=
  (slope_fun_def_field f _).symm ▸ fun _ hy _ hz hz' ↦ hfc.secant_mono hx (mem_of_mem_diff hy)
    (mem_of_mem_diff hz) (not_mem_of_mem_diff hy :) (not_mem_of_mem_diff hz :) hz'


/-- If `f : 𝕜 → 𝕜` is concave on `s`, then for any point `x ∈ s` the slope of the secant line of `f`
through `x` is antitone on `s \ {x}`. -/
lemma ConcaveOn.slope_anti (hfc : ConcaveOn 𝕜 s f) (hx : x ∈ s) :
    AntitoneOn (slope f x) (s \ {x}) := by
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    x : 𝕜
    hfc : ConcaveOn 𝕜 s f
    hx : Membership.mem s x
    ⊢ AntitoneOn (slope f x) (SDiff.sdiff s (Singleton.singleton x))
  -/
  rw [← neg_neg f, slope_neg_fun]
  /-
    𝕜 : Type u_1
    inst✝ : LinearOrderedField 𝕜
    s : Set 𝕜
    f : 𝕜 → 𝕜
    x : 𝕜
    hfc : ConcaveOn 𝕜 s f
    hx : Membership.mem s x
    ⊢ AntitoneOn (Neg.neg (slope (Neg.neg f)) x) (SDiff.sdiff s (Singleton.singlet …
  -/
  exact (ConvexOn.slope_mono hfc.neg hx).neg
  /-
    🎉 no goals
  -/


/-- If `f : ℝ → ℝ` is convex on `S` and right-differentiable at `x ∈ S`, then the slope of any
secant line with left endpoint at `x` is bounded below by the right derivative of `f` at `x`. -/
lemma le_slope_of_hasDerivWithinAt_Ioi (hfc : ConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Ioi x) x) :
    f' ≤ slope f x y := by
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    ⊢ LE.le f' (slope f x y)
  -/
  apply le_of_tendsto <| (hasDerivWithinAt_iff_tendsto_slope' not_mem_Ioi_self).mp hf'
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    ⊢ Filter.Eventually (fun c => LE.le (slope f x c) (slope f x y)) (nhdsWithin x …
  -/
  simp_rw [eventually_nhdsWithin_iff, slope_def_field]
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Ioi x) x_1 → LE.le (HDiv.h …
  -/
  filter_upwards [eventually_lt_nhds hxy] with t ht (ht' : x < t)
  /-
    case h
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    t : Real
    ht : LT.lt t y
    ht' : LT.lt x t
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f t) (f x)) (HSub.hSub t x)) (HDiv.hDiv (HSub.h …
  -/
  refine hfc.secant_mono hx (?_ : t ∈ S) hy ht'.ne' hxy.ne' ht.le
  /-
    case h
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    t : Real
    ht : LT.lt t y
    ht' : LT.lt x t
    ⊢ Membership.mem S t
  -/
  exact hfc.1.ordConnected.out hx hy ⟨ht'.le, ht.le⟩
  /-
    🎉 no goals
  -/


/-- Reformulation of `ConvexOn.le_slope_of_hasDerivWithinAt_Ioi` using `derivWithin`. -/
lemma right_deriv_le_slope (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Ioi x) x) :
    derivWithin f (Ioi x) x ≤ slope f x y :=
  le_slope_of_hasDerivWithinAt_Ioi hfc hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is convex on `S` and differentiable within `S` at `x`, then the slope of any
secant line with left endpoint at `x` is bounded below by the derivative of `f` within `S` at `x`.

This is fractionally weaker than `ConvexOn.le_slope_of_hasDerivWithinAt_Ioi` but simpler to apply
under a `DifferentiableOn S` hypothesis. -/
lemma le_slope_of_hasDerivWithinAt (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivWithinAt f f' S x) :
    f' ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt_Ioi hx hy hxy <|
    hf'.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsGT hx hy hxy


/-- Reformulation of `ConvexOn.le_slope_of_hasDerivWithinAt` using `derivWithin`. -/
lemma derivWithin_le_slope (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S x) :
    derivWithin f S x ≤ slope f x y :=
  le_slope_of_hasDerivWithinAt hfc hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is convex on `S` and differentiable at `x ∈ S`, then the slope of any secant
line with left endpoint at `x` is bounded below by the derivative of `f` at `x`. -/
lemma le_slope_of_hasDerivAt (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (ha : HasDerivAt f f' x) :
    f' ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt_Ioi hx hy hxy ha.hasDerivWithinAt


/-- Reformulation of `ConvexOn.le_slope_of_hasDerivAt` using `deriv` -/
lemma deriv_le_slope (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f x) :
    deriv f x ≤ slope f x y :=
  le_slope_of_hasDerivAt hfc hx hy hxy hfd.hasDerivAt


/-- If `f : ℝ → ℝ` is convex on `S` and left-differentiable at `y ∈ S`, then the slope of any secant
line with right endpoint at `y` is bounded above by the left derivative of `f` at `y`. -/
lemma slope_le_of_hasDerivWithinAt_Iio (hfc : ConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Iio y) y) :
    slope f x y ≤ f' := by
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    ⊢ LE.le (slope f x y) f'
  -/
  apply ge_of_tendsto <| (hasDerivWithinAt_iff_tendsto_slope' not_mem_Iio_self).mp hf'
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    ⊢ Filter.Eventually (fun c => LE.le (slope f x y) (slope f y c)) (nhdsWithin y …
  -/
  simp_rw [eventually_nhdsWithin_iff, slope_comm f x y, slope_def_field]
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    ⊢ Filter.Eventually (fun x_1 => Membership.mem (Set.Iio y) x_1 → LE.le (HDiv.h …
  -/
  filter_upwards [eventually_gt_nhds hxy] with t ht (ht' : t < y)
  /-
    case h
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    t : Real
    ht : LT.lt x t
    ht' : LT.lt t y
    ⊢ LE.le (HDiv.hDiv (HSub.hSub (f x) (f y)) (HSub.hSub x y)) (HDiv.hDiv (HSub.h …
  -/
  refine hfc.secant_mono hy hx (?_ : t ∈ S) hxy.ne ht'.ne ht.le
  /-
    case h
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : ConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    t : Real
    ht : LT.lt x t
    ht' : LT.lt t y
    ⊢ Membership.mem S t
  -/
  exact hfc.1.ordConnected.out hx hy ⟨ht.le, ht'.le⟩
  /-
    🎉 no goals
  -/


/-- Reformulation of `ConvexOn.slope_le_of_hasDerivWithinAt_Iio` using `derivWithin`. -/
lemma slope_le_left_deriv (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Iio y) y) :
    slope f x y ≤ derivWithin f (Iio y) y :=
  hfc.slope_le_of_hasDerivWithinAt_Iio hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is convex on `S` and differentiable within `S` at `y`, then the slope of any
secant line with right endpoint at `y` is bounded above by the derivative of `f` within `S` at `y`.

This is fractionally weaker than `ConvexOn.slope_le_of_hasDerivWithinAt_Iio` but simpler to apply
under a `DifferentiableOn S` hypothesis. -/
lemma slope_le_of_hasDerivWithinAt (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivWithinAt f f' S y) :
    slope f x y ≤ f' :=
  hfc.slope_le_of_hasDerivWithinAt_Iio hx hy hxy <|
    hf'.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsLT hx hy hxy


/-- Reformulation of `ConvexOn.slope_le_of_hasDerivWithinAt` using `derivWithin`. -/
lemma slope_le_derivWithin (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S y) :
    slope f x y ≤ derivWithin f S y :=
  hfc.slope_le_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is convex on `S` and differentiable at `y ∈ S`, then the slope of any secant
line with right endpoint at `y` is bounded above by the derivative of `f` at `y`. -/
lemma slope_le_of_hasDerivAt (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivAt f f' y) :
    slope f x y ≤ f' :=
  hfc.slope_le_of_hasDerivWithinAt_Iio hx hy hxy hf'.hasDerivWithinAt


/-- Reformulation of `ConvexOn.slope_le_of_hasDerivAt` using `deriv`. -/
lemma slope_le_deriv (hfc : ConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f y) :
    slope f x y ≤ deriv f y :=
  hfc.slope_le_of_hasDerivAt hx hy hxy hfd.hasDerivAt


/-- If `f` is convex on `S` and differentiable on `S`, then its derivative within `S` is monotone
on `S`. -/
lemma monotoneOn_derivWithin (hfc : ConvexOn ℝ S f) (hfd : DifferentiableOn ℝ f S) :
    MonotoneOn (derivWithin f S) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : ConvexOn Real S f
    hfd : DifferentiableOn Real f S
    ⊢ MonotoneOn (derivWithin f S) S
  -/
  intro x hx y hy hxy
  /-
    S : Set Real
    f : Real → Real
    hfc : ConvexOn Real S f
    hfd : DifferentiableOn Real f S
    x : Real
    hx : Membership.mem S x
    y : Real
    hy : Membership.mem S y
    hxy : LE.le x y
    ⊢ LE.le (derivWithin f S x) (derivWithin f S y)
  -/
  rcases eq_or_lt_of_le hxy with rfl | hxy'
    /-
      case inl
      S : Set Real
      f : Real → Real
      hfc : ConvexOn Real S f
      hfd : DifferentiableOn Real f S
      x : Real
      hx hy : Membership.mem S x
      hxy : LE.le x x
      ⊢ LE.le (derivWithin f S x) (derivWithin f S x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  exact (hfc.derivWithin_le_slope hx hy hxy' (hfd x hx)).trans
    (hfc.slope_le_derivWithin hx hy hxy' (hfd y hy))


/-- If `f` is convex on `S` and differentiable at all points of `S`, then its derivative is monotone
on `S`. -/
theorem monotoneOn_deriv (hfc : ConvexOn ℝ S f) (hfd : ∀ x ∈ S, DifferentiableAt ℝ f x) :
    MonotoneOn (deriv f) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : ConvexOn Real S f
    hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
    ⊢ MonotoneOn (deriv f) S
  -/
  intro x hx y hy hxy
  /-
    S : Set Real
    f : Real → Real
    hfc : ConvexOn Real S f
    hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
    x : Real
    hx : Membership.mem S x
    y : Real
    hy : Membership.mem S y
    hxy : LE.le x y
    ⊢ LE.le (deriv f x) (deriv f y)
  -/
  rcases eq_or_lt_of_le hxy with rfl | hxy'
    /-
      case inl
      S : Set Real
      f : Real → Real
      hfc : ConvexOn Real S f
      hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
      x : Real
      hx hy : Membership.mem S x
      hxy : LE.le x x
      ⊢ LE.le (deriv f x) (deriv f x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  /-
    case inr
    S : Set Real
    f : Real → Real
    hfc : ConvexOn Real S f
    hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
    x : Real
    hx : Membership.mem S x
    y : Real
    hy : Membership.mem S y
    hxy : LE.le x y
    hxy' : LT.lt x y
    ⊢ LE.le (deriv f x) (deriv f y)
  -/
  exact (hfc.deriv_le_slope hx hy hxy' (hfd x hx)).trans (hfc.slope_le_deriv hx hy hxy' (hfd y hy))
  /-
    🎉 no goals
  -/


/-- If `f : ℝ → ℝ` is strictly convex on `S` and right-differentiable at `x ∈ S`, then the slope of
any secant line with left endpoint at `x` is strictly greater than the right derivative of `f` at
`x`. -/
lemma lt_slope_of_hasDerivWithinAt_Ioi (hfc : StrictConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Ioi x) x) :
    f' < slope f x y := by
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    ⊢ LT.lt f' (slope f x y)
  -/
  obtain ⟨u, hxu, huy⟩ := exists_between hxy
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    ⊢ LT.lt f' (slope f x y)
  -/
  have hu : u ∈ S := hfc.1.ordConnected.out hx hy ⟨hxu.le, huy.le⟩
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    ⊢ LT.lt f' (slope f x y)
  -/
  have := hfc.secant_strict_mono hx hu hy hxu.ne' hxy.ne' huy
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    this : LT.lt (HDiv.hDiv (HSub.hSub (f u) (f x)) (HSub.hSub u x)) (HDiv.hDiv (H …
    ⊢ LT.lt f' (slope f x y)
  -/
  simp only [← slope_def_field] at this
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Ioi x) x
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    this : LT.lt (slope f x u) (slope f x y)
    ⊢ LT.lt f' (slope f x y)
  -/
  exact (hfc.convexOn.le_slope_of_hasDerivWithinAt_Ioi hx hu hxu hf').trans_lt this
  /-
    🎉 no goals
  -/


lemma right_deriv_lt_slope (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Ioi x) x) :
    derivWithin f (Ioi x) x < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt_Ioi hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is strictly convex on `S` and differentiable within `S` at `x ∈ S`, then the
slope of any secant line with left endpoint at `x` is strictly greater than the derivative of `f`
within `S` at `x`.

This is fractionally weaker than `StrictConvexOn.lt_slope_of_hasDerivWithinAt_Ioi` but simpler to
apply under a `DifferentiableOn S` hypothesis. -/
lemma lt_slope_of_hasDerivWithinAt (hfc : StrictConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' S x) :
    f' < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt_Ioi hx hy hxy <|
    hf'.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsGT hx hy hxy


lemma derivWithin_lt_slope (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S x) :
    derivWithin f S x < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is strictly convex on `S` and differentiable at `x ∈ S`, then the slope of any
secant line with left endpoint at `x` is strictly greater than the derivative of `f` at `x`. -/
lemma lt_slope_of_hasDerivAt (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivAt f f' x) :
    f' < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt_Ioi hx hy hxy hf'.hasDerivWithinAt


lemma deriv_lt_slope (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f x) :
    deriv f x < slope f x y :=
  hfc.lt_slope_of_hasDerivAt hx hy hxy hfd.hasDerivAt


/-- If `f : ℝ → ℝ` is strictly convex on `S` and differentiable at `y ∈ S`, then the slope of any
secant line with right endpoint at `y` is strictly less than the left derivative at `y`. -/
lemma slope_lt_of_hasDerivWithinAt_Iio (hfc : StrictConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Iio y) y)  :
    slope f x y < f' := by
  /-
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    ⊢ LT.lt (slope f x y) f'
  -/
  obtain ⟨u, hxu, huy⟩ := exists_between hxy
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    ⊢ LT.lt (slope f x y) f'
  -/
  have hu : u ∈ S := hfc.1.ordConnected.out hx hy ⟨hxu.le, huy.le⟩
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    ⊢ LT.lt (slope f x y) f'
  -/
  have := hfc.secant_strict_mono hy hx hu hxy.ne huy.ne hxu
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    this : LT.lt (HDiv.hDiv (HSub.hSub (f x) (f y)) (HSub.hSub x y)) (HDiv.hDiv (H …
    ⊢ LT.lt (slope f x y) f'
  -/
  simp_rw [← slope_def_field, slope_comm _ y] at this
  /-
    case intro.intro
    S : Set Real
    f : Real → Real
    x y f' : Real
    hfc : StrictConvexOn Real S f
    hx : Membership.mem S x
    hy : Membership.mem S y
    hxy : LT.lt x y
    hf' : HasDerivWithinAt f f' (Set.Iio y) y
    u : Real
    hxu : LT.lt x u
    huy : LT.lt u y
    hu : Membership.mem S u
    this : LT.lt (slope f x y) (slope f u y)
    ⊢ LT.lt (slope f x y) f'
  -/
  exact this.trans_le <| hfc.convexOn.slope_le_of_hasDerivWithinAt_Iio hu hy huy hf'
  /-
    🎉 no goals
  -/


lemma slope_lt_left_deriv (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Iio y) y)  :
    slope f x y < derivWithin f (Iio y) y :=
  hfc.slope_lt_of_hasDerivWithinAt_Iio hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is strictly convex on `S` and differentiable within `S` at `y ∈ S`, then the
slope of any secant line with right endpoint at `y` is strictly less than the derivative of `f`
within `S` at `y`.

This is fractionally weaker than `StrictConvexOn.slope_lt_of_hasDerivWithinAt_Iio` but simpler to
apply under a `DifferentiableOn S` hypothesis.-/
lemma slope_lt_of_hasDerivWithinAt (hfc : StrictConvexOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' S y) :
    slope f x y < f' :=
  hfc.slope_lt_of_hasDerivWithinAt_Iio hx hy hxy <|
    hf'.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsLT hx hy hxy


lemma slope_lt_derivWithin (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S y) :
    slope f x y < derivWithin f S y :=
  hfc.slope_lt_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


/-- If `f : ℝ → ℝ` is strictly convex on `S` and differentiable at `y ∈ S`, then the slope of any
secant line with right endpoint at `y` is strictly less than the derivative of `f` at `y`. -/
lemma slope_lt_of_hasDerivAt (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivAt f f' y) :
    slope f x y < f' :=
  hfc.slope_lt_of_hasDerivWithinAt_Iio hx hy hxy hf'.hasDerivWithinAt


lemma slope_lt_deriv (hfc : StrictConvexOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f y) :
    slope f x y < deriv f y :=
  hfc.slope_lt_of_hasDerivAt hx hy hxy hfd.hasDerivAt


/-- If `f` is convex on `S` and differentiable on `S`, then its derivative within `S` is monotone
on `S`. -/
lemma strictMonoOn_derivWithin (hfc : StrictConvexOn ℝ S f) (hfd : DifferentiableOn ℝ f S) :
    StrictMonoOn (derivWithin f S) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : StrictConvexOn Real S f
    hfd : DifferentiableOn Real f S
    ⊢ StrictMonoOn (derivWithin f S) S
  -/
  intro x hx y hy hxy
  exact (hfc.derivWithin_lt_slope hx hy hxy (hfd x hx)).trans
    (hfc.slope_lt_derivWithin hx hy hxy (hfd y hy))


/-- If `f` is convex on `S` and differentiable at all points of `S`, then its derivative is monotone
on `S`. -/
lemma strictMonoOn_deriv (hfc : StrictConvexOn ℝ S f) (hfd : ∀ x ∈ S, DifferentiableAt ℝ f x) :
    StrictMonoOn (deriv f) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : StrictConvexOn Real S f
    hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
    ⊢ StrictMonoOn (deriv f) S
  -/
  intro x hx y hy hxy
  /-
    S : Set Real
    f : Real → Real
    hfc : StrictConvexOn Real S f
    hfd : ∀ (x : Real), Membership.mem S x → DifferentiableAt Real f x
    x : Real
    hx : Membership.mem S x
    y : Real
    hy : Membership.mem S y
    hxy : LT.lt x y
    ⊢ LT.lt (deriv f x) (deriv f y)
  -/
  exact (hfc.deriv_lt_slope hx hy hxy (hfd x hx)).trans (hfc.slope_lt_deriv hx hy hxy (hfd y hy))
  /-
    🎉 no goals
  -/


lemma slope_le_of_hasDerivWithinAt_Ioi (hfc : ConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Ioi x) x) :
    slope f x y ≤ f' := by
  simpa only [Pi.neg_def, slope_neg, neg_neg] using
    neg_le_neg (hfc.neg.le_slope_of_hasDerivWithinAt_Ioi hx hy hxy hf'.neg)


lemma slope_le_right_deriv (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Ioi x) x) :
    slope f x y ≤ derivWithin f (Ioi x) x :=
  hfc.slope_le_of_hasDerivWithinAt_Ioi hx hy hxy hfd.hasDerivWithinAt


lemma slope_le_of_hasDerivWithinAt (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : HasDerivWithinAt f f' S x) :
    slope f x y ≤ f' :=
  hfc.slope_le_of_hasDerivWithinAt_Ioi hx hy hxy <|
    hfd.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsGT hx hy hxy


lemma slope_le_derivWithin (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S x) :
    slope f x y ≤ derivWithin f S x :=
  hfc.slope_le_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


lemma slope_le_of_hasDerivAt (hfc : ConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivAt f f' x) :
    slope f x y ≤ f' :=
  hfc.slope_le_of_hasDerivWithinAt_Ioi hx hy hxy hf'.hasDerivWithinAt


lemma slope_le_deriv (hfc : ConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hfd : DifferentiableAt ℝ f x) :
    slope f x y ≤ deriv f x :=
  hfc.slope_le_of_hasDerivAt hx hy hxy hfd.hasDerivAt


lemma le_slope_of_hasDerivWithinAt_Iio (hfc : ConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Iio y) y) :
    f' ≤ slope f x y := by
  simpa only [neg_neg, Pi.neg_def, slope_neg] using
    neg_le_neg (hfc.neg.slope_le_of_hasDerivWithinAt_Iio hx hy hxy hf'.neg)


lemma left_deriv_le_slope (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Iio y) y) :
    derivWithin f (Iio y) y ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt_Iio hx hy hxy hfd.hasDerivWithinAt


lemma le_slope_of_hasDerivWithinAt (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivWithinAt f f' S y) :
    f' ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt_Iio hx hy hxy <|
    hf'.mono_of_mem_nhdsWithin <| hfc.1.ordConnected.mem_nhdsLT hx hy hxy


lemma derivWithin_le_slope (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S y) :
    derivWithin f S y ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


lemma le_slope_of_hasDerivAt (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivAt f f' y) :
    f' ≤ slope f x y :=
  hfc.le_slope_of_hasDerivWithinAt_Iio hx hy hxy hf'.hasDerivWithinAt


lemma deriv_le_slope (hfc : ConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f y) :
    deriv f y ≤ slope f x y :=
  hfc.le_slope_of_hasDerivAt hx hy hxy hfd.hasDerivAt


lemma antitoneOn_derivWithin (hfc : ConcaveOn ℝ S f) (hfd : DifferentiableOn ℝ f S) :
    AntitoneOn (derivWithin f S) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : ConcaveOn Real S f
    hfd : DifferentiableOn Real f S
    ⊢ AntitoneOn (derivWithin f S) S
  -/
  intro x hx y hy hxy
  /-
    S : Set Real
    f : Real → Real
    hfc : ConcaveOn Real S f
    hfd : DifferentiableOn Real f S
    x : Real
    hx : Membership.mem S x
    y : Real
    hy : Membership.mem S y
    hxy : LE.le x y
    ⊢ LE.le (derivWithin f S y) (derivWithin f S x)
  -/
  rcases eq_or_lt_of_le hxy with rfl | hxy'
    /-
      case inl
      S : Set Real
      f : Real → Real
      hfc : ConcaveOn Real S f
      hfd : DifferentiableOn Real f S
      x : Real
      hx hy : Membership.mem S x
      hxy : LE.le x x
      ⊢ LE.le (derivWithin f S x) (derivWithin f S x)
    -/
  · rfl
    /-
      🎉 no goals
    -/
  exact (hfc.derivWithin_le_slope hx hy hxy' (hfd y hy)).trans
    (hfc.slope_le_derivWithin hx hy hxy' (hfd x hx))


/-- If `f` is concave on `S` and differentiable at all points of `S`, then its derivative is
antitone (monotone decreasing) on `S`. -/
theorem antitoneOn_deriv (hfc : ConcaveOn ℝ S f) (hfd : ∀ x ∈ S, DifferentiableAt ℝ f x) :
    AntitoneOn (deriv f) S := by
  simpa only [Pi.neg_def, deriv.neg, neg_neg] using
    (hfc.neg.monotoneOn_deriv (fun x hx ↦ (hfd x hx).neg)).neg


lemma slope_lt_of_hasDerivWithinAt_Ioi (hfc : StrictConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Ioi x) x) :
    slope f x y < f' := by
  simpa only [Pi.neg_def, slope_neg, neg_neg] using
    neg_lt_neg (hfc.neg.lt_slope_of_hasDerivWithinAt_Ioi hx hy hxy hf'.neg)


lemma slope_lt_right_deriv (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Ioi x) x) :
    slope f x y < derivWithin f (Ioi x) x :=
  hfc.slope_lt_of_hasDerivWithinAt_Ioi hx hy hxy hfd.hasDerivWithinAt


lemma slope_lt_of_hasDerivWithinAt (hfc : StrictConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hfd : HasDerivWithinAt f f' S x) :
    slope f x y < f' := by
  simpa only [Pi.neg_def, slope_neg, neg_neg] using
    neg_lt_neg (hfc.neg.lt_slope_of_hasDerivWithinAt hx hy hxy hfd.neg)


lemma slope_lt_derivWithin (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S x) :
    slope f x y < derivWithin f S x :=
  hfc.slope_lt_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


lemma slope_lt_of_hasDerivAt (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : HasDerivAt f f' x) :
    slope f x y < f' := by
  simpa only [Pi.neg_def, slope_neg, neg_neg] using
    neg_lt_neg (hfc.neg.lt_slope_of_hasDerivAt hx hy hxy hfd.neg)


lemma slope_lt_deriv (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f x) :
    slope f x y < deriv f x :=
  hfc.slope_lt_of_hasDerivAt hx hy hxy hfd.hasDerivAt


lemma lt_slope_of_hasDerivWithinAt_Iio (hfc : StrictConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' (Iio y) y) :
    f' < slope f x y := by
  simpa only [Pi.neg_def, slope_neg, neg_neg] using
    neg_lt_neg (hfc.neg.slope_lt_of_hasDerivWithinAt_Iio hx hy hxy hf'.neg)


lemma left_deriv_lt_slope (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f (Iio y) y) :
    derivWithin f (Iio y) y < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt_Iio hx hy hxy hfd.hasDerivWithinAt


lemma lt_slope_of_hasDerivWithinAt (hfc : StrictConcaveOn ℝ S f)
    (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y) (hf' : HasDerivWithinAt f f' S y) :
    f' < slope f x y := by
  simpa only [neg_neg, Pi.neg_def, slope_neg] using
    neg_lt_neg (hfc.neg.slope_lt_of_hasDerivWithinAt hx hy hxy hf'.neg)


lemma derivWithin_lt_slope (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableWithinAt ℝ f S y) :
    derivWithin f S y < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt hx hy hxy hfd.hasDerivWithinAt


lemma lt_slope_of_hasDerivAt (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hf' : HasDerivAt f f' y) :
    f' < slope f x y :=
  hfc.lt_slope_of_hasDerivWithinAt_Iio hx hy hxy hf'.hasDerivWithinAt


lemma deriv_lt_slope (hfc : StrictConcaveOn ℝ S f) (hx : x ∈ S) (hy : y ∈ S) (hxy : x < y)
    (hfd : DifferentiableAt ℝ f y) :
    deriv f y < slope f x y :=
  hfc.lt_slope_of_hasDerivAt hx hy hxy hfd.hasDerivAt


lemma strictAntiOn_derivWithin (hfc : StrictConcaveOn ℝ S f) (hfd : DifferentiableOn ℝ f S) :
    StrictAntiOn (derivWithin f S) S := by
  /-
    S : Set Real
    f : Real → Real
    hfc : StrictConcaveOn Real S f
    hfd : DifferentiableOn Real f S
    ⊢ StrictAntiOn (derivWithin f S) S
  -/
  intro x hx y hy hxy
  exact (hfc.derivWithin_lt_slope hx hy hxy (hfd y hy)).trans
    (hfc.slope_lt_derivWithin hx hy hxy (hfd x hx))


theorem strictAntiOn_deriv (hfc : StrictConcaveOn ℝ S f) (hfd : ∀ x ∈ S, DifferentiableAt ℝ f x) :
    StrictAntiOn (deriv f) S := by
  simpa only [Pi.neg_def, deriv.neg, neg_neg] using
    (hfc.neg.strictMonoOn_deriv (fun x hx ↦ (hfd x hx).neg)).neg


