/-- `f` has constant speed `l` on `s` if the variation of `f` on `s ∩ Icc x y` is equal to
`l * (y - x)` for any `x y` in `s`.
-/
def HasConstantSpeedOnWith :=
  ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), eVariationOn f (s ∩ Icc x y) = ENNReal.ofReal (l * (y - x))


theorem HasConstantSpeedOnWith.hasLocallyBoundedVariationOn (h : HasConstantSpeedOnWith f s l) :
    LocallyBoundedVariationOn f s := fun x y hx hy => by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    h : HasConstantSpeedOnWith f s l
    x y : Real
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ BoundedVariationOn f (Inter.inter s (Set.Icc x y))
  -/
  simp only [BoundedVariationOn, h hx hy, Ne, ENNReal.ofReal_ne_top, not_false_iff]
  /-
    🎉 no goals
  -/


theorem hasConstantSpeedOnWith_of_subsingleton (f : ℝ → E) {s : Set ℝ} (hs : s.Subsingleton)
    (l : ℝ≥0) : HasConstantSpeedOnWith f s l := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    hs : s.Subsingleton
    l : NNReal
    ⊢ HasConstantSpeedOnWith f s l
  -/
  rintro x hx y hy; cases hs hx hy
  /-
    case refl
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    hs : s.Subsingleton
    l : NNReal
    x : Real
    hx hy : Membership.mem s x
    ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x x))) (ENNReal.ofReal (HMul.hMul …
  -/
  rw [eVariationOn.subsingleton f (fun y hy z hz => hs hy.1 hz.1 : (s ∩ Icc x x).Subsingleton)]
  /-
    case refl
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    hs : s.Subsingleton
    l : NNReal
    x : Real
    hx hy : Membership.mem s x
    ⊢ Eq 0 (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x x)))
  -/
  simp only [sub_self, mul_zero, ENNReal.ofReal_zero]
  /-
    🎉 no goals
  -/


theorem hasConstantSpeedOnWith_iff_ordered :
    HasConstantSpeedOnWith f s l ↔ ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s),
      x ≤ y → eVariationOn f (s ∩ Icc x y) = ENNReal.ofReal (l * (y - x)) := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    ⊢ Iff (HasConstantSpeedOnWith f s l) (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y  …
  -/
  refine ⟨fun h x xs y ys _ => h xs ys, fun h x xs y ys => ?_⟩
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
    x : Real
    xs : Membership.mem s x
    y : Real
    ys : Membership.mem s y
    ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x y))) (ENNReal.ofReal (HMul.hMul …
  -/
  rcases le_total x y with (xy | yx)
    /-
      case inl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      xy : LE.le x y
      ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x y))) (ENNReal.ofReal (HMul.hMul …
    -/
  · exact h xs ys xy
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      yx : LE.le y x
      ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x y))) (ENNReal.ofReal (HMul.hMul …
    -/
  · rw [eVariationOn.subsingleton, ENNReal.ofReal_of_nonpos]
      /-
        case inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        ⊢ LE.le (HMul.hMul (↑l) (HSub.hSub y x)) 0
      -/
    · exact mul_nonpos_of_nonneg_of_nonpos l.prop (sub_nonpos_of_le yx)
      /-
        🎉 no goals
      -/
      /-
        case inr.hs
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        ⊢ (Inter.inter s (Set.Icc x y)).Subsingleton
      -/
    · rintro z ⟨zs, xz, zy⟩ w ⟨ws, xw, wy⟩
      /-
        case inr.hs.intro.intro.intro.intro
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        z : Real
        zs : Membership.mem s z
        xz : LE.le x z
        zy : LE.le z y
        w : Real
        ws : Membership.mem s w
        xw : LE.le x w
        wy : LE.le w y
        ⊢ Eq z w
      -/
      cases le_antisymm (zy.trans yx) xz
      /-
        case inr.hs.intro.intro.intro.intro.refl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        w : Real
        ws : Membership.mem s w
        xw : LE.le x w
        wy : LE.le w y
        zs : Membership.mem s x
        xz : LE.le x x
        zy : LE.le x y
        ⊢ Eq x w
      -/
      cases le_antisymm (wy.trans yx) xw
      /-
        case inr.hs.intro.intro.intro.intro.refl.refl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        zs : Membership.mem s x
        xz : LE.le x x
        zy : LE.le x y
        ws : Membership.mem s x
        xw : LE.le x x
        wy : LE.le x y
        ⊢ Eq x x
      -/
      rfl
      /-
        🎉 no goals
      -/


theorem hasConstantSpeedOnWith_iff_variationOnFromTo_eq :
    HasConstantSpeedOnWith f s l ↔ LocallyBoundedVariationOn f s ∧
      ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), variationOnFromTo f s x y = l * (y - x) := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    ⊢ Iff (HasConstantSpeedOnWith f s l) (And (LocallyBoundedVariationOn f s) (∀ ⦃ …
  -/
  constructor
    /-
      case mp
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      ⊢ HasConstantSpeedOnWith f s l → And (LocallyBoundedVariationOn f s) (∀ ⦃x : R …
    -/
  · rintro h; refine ⟨h.hasLocallyBoundedVariationOn, fun x xs y ys => ?_⟩
    /-
      case mp
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      h : HasConstantSpeedOnWith f s l
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      ⊢ Eq (variationOnFromTo f s x y) (HMul.hMul (↑l) (HSub.hSub y x))
    -/
    rw [hasConstantSpeedOnWith_iff_ordered] at h
    /-
      case mp
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      ⊢ Eq (variationOnFromTo f s x y) (HMul.hMul (↑l) (HSub.hSub y x))
    -/
    rcases le_total x y with (xy | yx)
      /-
        case mp.inl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        xy : LE.le x y
        ⊢ Eq (variationOnFromTo f s x y) (HMul.hMul (↑l) (HSub.hSub y x))
      -/
    · rw [variationOnFromTo.eq_of_le f s xy, h xs ys xy]
      /-
        case mp.inl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        xy : LE.le x y
        ⊢ Eq (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub y x))).toReal (HMul.hMul (↑l)  …
      -/
      exact ENNReal.toReal_ofReal (mul_nonneg l.prop (sub_nonneg.mpr xy))
      /-
        🎉 no goals
      -/
      /-
        case mp.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        ⊢ Eq (variationOnFromTo f s x y) (HMul.hMul (↑l) (HSub.hSub y x))
      -/
    · rw [variationOnFromTo.eq_of_ge f s yx, h ys xs yx]
      /-
        case mp.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        ⊢ Eq (Neg.neg (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x y))).toReal) (HMul. …
      -/
      have := ENNReal.toReal_ofReal (mul_nonneg l.prop (sub_nonneg.mpr yx))
      /-
        case mp.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE.l …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        yx : LE.le y x
        this : Eq (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x y))).toReal (HMul.hMul  …
        ⊢ Eq (Neg.neg (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x y))).toReal) (HMul. …
      -/
      simp_all only [NNReal.val_eq_coe]; ring
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      ⊢ And (LocallyBoundedVariationOn f s) (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y …
    -/
  · rw [hasConstantSpeedOnWith_iff_ordered]
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      ⊢ And (LocallyBoundedVariationOn f s) (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y …
    -/
    rintro h x xs y ys xy
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      h : And (LocallyBoundedVariationOn f s) (∀ ⦃x : Real⦄, Membership.mem s x → ∀  …
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      xy : LE.le x y
      ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x y))) (ENNReal.ofReal (HMul.hMul …
    -/
    rw [← h.2 xs ys, variationOnFromTo.eq_of_le f s xy, ENNReal.ofReal_toReal (h.1 x y xs ys)]
    /-
      🎉 no goals
    -/


theorem HasConstantSpeedOnWith.union {t : Set ℝ} (hfs : HasConstantSpeedOnWith f s l)
    (hft : HasConstantSpeedOnWith f t l) {x : ℝ} (hs : IsGreatest s x) (ht : IsLeast t x) :
    HasConstantSpeedOnWith f (s ∪ t) l := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    t : Set Real
    hfs : HasConstantSpeedOnWith f s l
    hft : HasConstantSpeedOnWith f t l
    x : Real
    hs : IsGreatest s x
    ht : IsLeast t x
    ⊢ HasConstantSpeedOnWith f (Union.union s t) l
  -/
  rw [hasConstantSpeedOnWith_iff_ordered] at hfs hft ⊢
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l : NNReal
    t : Set Real
    hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
    hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
    x : Real
    hs : IsGreatest s x
    ht : IsLeast t x
    ⊢ ∀ ⦃x : Real⦄, Membership.mem (Union.union s t) x → ∀ ⦃y : Real⦄, Membership. …
  -/
  rintro z (zs | zt) y (ys | yt) zy
  · have : (s ∪ t) ∩ Icc z y = s ∩ Icc z y := by
      ext w; constructor
      · rintro ⟨ws | wt, zw, wy⟩
        · exact ⟨ws, zw, wy⟩
        · exact ⟨(le_antisymm (wy.trans (hs.2 ys)) (ht.2 wt)).symm ▸ hs.1, zw, wy⟩
      · rintro ⟨ws, zwy⟩; exact ⟨Or.inl ws, zwy⟩
    /-
      case inl.inl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zs : Membership.mem s z
      y : Real
      ys : Membership.mem s y
      zy : LE.le z y
      this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Inter.inter s (Set.Ic …
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Set.Icc z y))) (ENNReal.o …
    -/
    rw [this, hfs zs ys zy]
    /-
      🎉 no goals
    -/
  · have : (s ∪ t) ∩ Icc z y = s ∩ Icc z x ∪ t ∩ Icc x y := by
      ext w; constructor
      · rintro ⟨ws | wt, zw, wy⟩
        exacts [Or.inl ⟨ws, zw, hs.2 ws⟩, Or.inr ⟨wt, ht.2 wt, wy⟩]
      · rintro (⟨ws, zw, wx⟩ | ⟨wt, xw, wy⟩)
        exacts [⟨Or.inl ws, zw, wx.trans (ht.2 yt)⟩, ⟨Or.inr wt, (hs.2 zs).trans xw, wy⟩]
    /-
      case inl.inr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zs : Membership.mem s z
      y : Real
      yt : Membership.mem t y
      zy : LE.le z y
      this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Union.union (Inter.in …
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Set.Icc z y))) (ENNReal.o …
    -/
    rw [this, @eVariationOn.union _ _ _ _ f _ _ x, hfs zs hs.1 (hs.2 zs), hft ht.1 yt (ht.2 yt)]
    · have q := ENNReal.ofReal_add (mul_nonneg l.prop (sub_nonneg.mpr (hs.2 zs)))
        (mul_nonneg l.prop (sub_nonneg.mpr (ht.2 yt)))
      /-
        case inl.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        t : Set Real
        hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
        hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
        x : Real
        hs : IsGreatest s x
        ht : IsLeast t x
        z : Real
        zs : Membership.mem s z
        y : Real
        yt : Membership.mem t y
        zy : LE.le z y
        this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Union.union (Inter.in …
        q : Eq (ENNReal.ofReal (HAdd.hAdd (HMul.hMul (↑l) (HSub.hSub x z)) (HMul.hMul  …
        ⊢ Eq (HAdd.hAdd (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x z))) (ENNReal.ofR …
      -/
      simp only [NNReal.val_eq_coe] at q
      /-
        case inl.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        t : Set Real
        hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
        hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
        x : Real
        hs : IsGreatest s x
        ht : IsLeast t x
        z : Real
        zs : Membership.mem s z
        y : Real
        yt : Membership.mem t y
        zy : LE.le z y
        this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Union.union (Inter.in …
        q : Eq (ENNReal.ofReal (HAdd.hAdd (HMul.hMul (↑l) (HSub.hSub x z)) (HMul.hMul  …
        ⊢ Eq (HAdd.hAdd (ENNReal.ofReal (HMul.hMul (↑l) (HSub.hSub x z))) (ENNReal.ofR …
      -/
      rw [← q]
      /-
        case inl.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        l : NNReal
        t : Set Real
        hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
        hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
        x : Real
        hs : IsGreatest s x
        ht : IsLeast t x
        z : Real
        zs : Membership.mem s z
        y : Real
        yt : Membership.mem t y
        zy : LE.le z y
        this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Union.union (Inter.in …
        q : Eq (ENNReal.ofReal (HAdd.hAdd (HMul.hMul (↑l) (HSub.hSub x z)) (HMul.hMul  …
        ⊢ Eq (ENNReal.ofReal (HAdd.hAdd (HMul.hMul (↑l) (HSub.hSub x z)) (HMul.hMul (↑ …
      -/
      ring_nf
      /-
        🎉 no goals
      -/
    exacts [⟨⟨hs.1, hs.2 zs, le_rfl⟩, fun w ⟨_, _, wx⟩ => wx⟩,
      ⟨⟨ht.1, le_rfl, ht.2 yt⟩, fun w ⟨_, xw, _⟩ => xw⟩]
    /-
      case inr.inl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zt : Membership.mem t z
      y : Real
      ys : Membership.mem s y
      zy : LE.le z y
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Set.Icc z y))) (ENNReal.o …
    -/
  · cases le_antisymm zy ((hs.2 ys).trans (ht.2 zt))
    /-
      case inr.inl.refl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zt : Membership.mem t z
      ys : Membership.mem s z
      zy : LE.le z z
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Set.Icc z z))) (ENNReal.o …
    -/
    simp only [Icc_self, sub_self, mul_zero, ENNReal.ofReal_zero]
    /-
      case inr.inl.refl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zt : Membership.mem t z
      ys : Membership.mem s z
      zy : LE.le z z
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Singleton.singleton z))) 0
    -/
    exact eVariationOn.subsingleton _ fun _ ⟨_, uz⟩ _ ⟨_, vz⟩ => uz.trans vz.symm
    /-
      🎉 no goals
    -/
  · have : (s ∪ t) ∩ Icc z y = t ∩ Icc z y := by
      ext w; constructor
      · rintro ⟨ws | wt, zw, wy⟩
        · exact ⟨le_antisymm ((ht.2 zt).trans zw) (hs.2 ws) ▸ ht.1, zw, wy⟩
        · exact ⟨wt, zw, wy⟩
      · rintro ⟨wt, zwy⟩; exact ⟨Or.inr wt, zwy⟩
    /-
      case inr.inr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      l : NNReal
      t : Set Real
      hfs : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → LE …
      hft : ∀ ⦃x : Real⦄, Membership.mem t x → ∀ ⦃y : Real⦄, Membership.mem t y → LE …
      x : Real
      hs : IsGreatest s x
      ht : IsLeast t x
      z : Real
      zt : Membership.mem t z
      y : Real
      yt : Membership.mem t y
      zy : LE.le z y
      this : Eq (Inter.inter (Union.union s t) (Set.Icc z y)) (Inter.inter t (Set.Ic …
      ⊢ Eq (eVariationOn f (Inter.inter (Union.union s t) (Set.Icc z y))) (ENNReal.o …
    -/
    rw [this, hft zt yt zy]
    /-
      🎉 no goals
    -/


theorem HasConstantSpeedOnWith.Icc_Icc {x y z : ℝ} (hfs : HasConstantSpeedOnWith f (Icc x y) l)
    (hft : HasConstantSpeedOnWith f (Icc y z) l) : HasConstantSpeedOnWith f (Icc x z) l := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    l : NNReal
    x y z : Real
    hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
    hft : HasConstantSpeedOnWith f (Set.Icc y z) l
    ⊢ HasConstantSpeedOnWith f (Set.Icc x z) l
  -/
  rcases le_total x y with (xy | yx)
    /-
      case inl
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      l : NNReal
      x y z : Real
      hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
      hft : HasConstantSpeedOnWith f (Set.Icc y z) l
      xy : LE.le x y
      ⊢ HasConstantSpeedOnWith f (Set.Icc x z) l
    -/
  · rcases le_total y z with (yz | zy)
      /-
        case inl.inl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        l : NNReal
        x y z : Real
        hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
        hft : HasConstantSpeedOnWith f (Set.Icc y z) l
        xy : LE.le x y
        yz : LE.le y z
        ⊢ HasConstantSpeedOnWith f (Set.Icc x z) l
      -/
    · rw [← Set.Icc_union_Icc_eq_Icc xy yz]
      /-
        case inl.inl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        l : NNReal
        x y z : Real
        hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
        hft : HasConstantSpeedOnWith f (Set.Icc y z) l
        xy : LE.le x y
        yz : LE.le y z
        ⊢ HasConstantSpeedOnWith f (Union.union (Set.Icc x y) (Set.Icc y z)) l
      -/
      exact hfs.union hft (isGreatest_Icc xy) (isLeast_Icc yz)
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        l : NNReal
        x y z : Real
        hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
        hft : HasConstantSpeedOnWith f (Set.Icc y z) l
        xy : LE.le x y
        zy : LE.le z y
        ⊢ HasConstantSpeedOnWith f (Set.Icc x z) l
      -/
    · rintro u ⟨xu, uz⟩ v ⟨xv, vz⟩
      rw [Icc_inter_Icc, sup_of_le_right xu, inf_of_le_right vz, ←
        hfs ⟨xu, uz.trans zy⟩ ⟨xv, vz.trans zy⟩, Icc_inter_Icc, sup_of_le_right xu,
        inf_of_le_right (vz.trans zy)]
    /-
      case inr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      l : NNReal
      x y z : Real
      hfs : HasConstantSpeedOnWith f (Set.Icc x y) l
      hft : HasConstantSpeedOnWith f (Set.Icc y z) l
      yx : LE.le y x
      ⊢ HasConstantSpeedOnWith f (Set.Icc x z) l
    -/
  · rintro u ⟨xu, uz⟩ v ⟨xv, vz⟩
    rw [Icc_inter_Icc, sup_of_le_right xu, inf_of_le_right vz, ←
      hft ⟨yx.trans xu, uz⟩ ⟨yx.trans xv, vz⟩, Icc_inter_Icc, sup_of_le_right (yx.trans xu),
      inf_of_le_right vz]


theorem hasConstantSpeedOnWith_zero_iff :
    HasConstantSpeedOnWith f s 0 ↔ ∀ᵉ (x ∈ s) (y ∈ s), edist (f x) (f y) = 0 := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    ⊢ Iff (HasConstantSpeedOnWith f s 0) (∀ (x : Real), Membership.mem s x → ∀ (y  …
  -/
  dsimp [HasConstantSpeedOnWith]
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    ⊢ Iff (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → E …
  -/
  simp only [zero_mul, ENNReal.ofReal_zero, ← eVariationOn.eq_zero_iff]
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    ⊢ Iff (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → E …
  -/
  constructor
    /-
      case mp
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      ⊢ (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → Eq (e …
    -/
  · by_contra!
    /-
      case mp
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      this : And (∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s  …
      ⊢ False
    -/
    obtain ⟨h, hfs⟩ := this
    /-
      case mp.intro
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → Eq ( …
      hfs : Ne (eVariationOn f s) 0
      ⊢ False
    -/
    simp_rw [ne_eq, eVariationOn.eq_zero_iff] at hfs h
    /-
      case mp.intro
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      hfs : Not (∀ (x : Real), Membership.mem s x → ∀ (y : Real), Membership.mem s y …
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
      ⊢ False
    -/
    push_neg at hfs
    /-
      case mp.intro
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
      hfs : Exists fun x => And (Membership.mem s x) (Exists fun y => And (Membershi …
      ⊢ False
    -/
    obtain ⟨x, xs, y, ys, hxy⟩ := hfs
    /-
      case mp.intro.intro.intro.intro.intro
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
      x : Real
      xs : Membership.mem s x
      y : Real
      ys : Membership.mem s y
      hxy : Ne (EDist.edist (f x) (f y)) 0
      ⊢ False
    -/
    rcases le_total x y with (xy | yx)
      /-
        case mp.intro.intro.intro.intro.intro.inl
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        hxy : Ne (EDist.edist (f x) (f y)) 0
        xy : LE.le x y
        ⊢ False
      -/
    · exact hxy (h xs ys x ⟨xs, le_rfl, xy⟩ y ⟨ys, xy, le_rfl⟩)
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.intro.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        hxy : Ne (EDist.edist (f x) (f y)) 0
        yx : LE.le y x
        ⊢ False
      -/
    · rw [edist_comm] at hxy
      /-
        case mp.intro.intro.intro.intro.intro.inr
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : Real → E
        s : Set Real
        h : ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, Membership.mem s y → ∀ (x …
        x : Real
        xs : Membership.mem s x
        y : Real
        ys : Membership.mem s y
        hxy : Ne (EDist.edist (f y) (f x)) 0
        yx : LE.le y x
        ⊢ False
      -/
      exact hxy (h ys xs y ⟨ys, le_rfl, yx⟩ x ⟨xs, yx, le_rfl⟩)
      /-
        🎉 no goals
      -/
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      ⊢ Eq (eVariationOn f s) 0 → ∀ ⦃x : Real⦄, Membership.mem s x → ∀ ⦃y : Real⦄, M …
    -/
  · rintro h x _ y _
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : Eq (eVariationOn f s) 0
      x : Real
      x✝¹ : Membership.mem s x
      y : Real
      x✝ : Membership.mem s y
      ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc x y))) 0
    -/
    refine le_antisymm ?_ zero_le'
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : Eq (eVariationOn f s) 0
      x : Real
      x✝¹ : Membership.mem s x
      y : Real
      x✝ : Membership.mem s y
      ⊢ LE.le (eVariationOn f (Inter.inter s (Set.Icc x y))) 0
    -/
    rw [← h]
    /-
      case mpr
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : Real → E
      s : Set Real
      h : Eq (eVariationOn f s) 0
      x : Real
      x✝¹ : Membership.mem s x
      y : Real
      x✝ : Membership.mem s y
      ⊢ LE.le (eVariationOn f (Inter.inter s (Set.Icc x y))) (eVariationOn f s)
    -/
    exact eVariationOn.mono f inter_subset_left
    /-
      🎉 no goals
    -/


theorem HasConstantSpeedOnWith.ratio {l' : ℝ≥0} (hl' : l' ≠ 0) {φ : ℝ → ℝ} (φm : MonotoneOn φ s)
    (hfφ : HasConstantSpeedOnWith (f ∘ φ) s l) (hf : HasConstantSpeedOnWith f (φ '' s) l') ⦃x : ℝ⦄
    (xs : x ∈ s) : EqOn φ (fun y => l / l' * (y - x) + φ x) s := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l l' : NNReal
    hl' : Ne l' 0
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s l
    hf : HasConstantSpeedOnWith f (Set.image φ s) l'
    x : Real
    xs : Membership.mem s x
    ⊢ Set.EqOn φ (fun y => HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑l ↑l') (HSub.hSub y x) …
  -/
  rintro y ys
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l l' : NNReal
    hl' : Ne l' 0
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s l
    hf : HasConstantSpeedOnWith f (Set.image φ s) l'
    x : Real
    xs : Membership.mem s x
    y : Real
    ys : Membership.mem s y
    ⊢ Eq (φ y) ((fun y => HAdd.hAdd (HMul.hMul (HDiv.hDiv ↑l ↑l') (HSub.hSub y x)) …
  -/
  rw [← sub_eq_iff_eq_add, mul_comm, ← mul_div_assoc, eq_div_iff (NNReal.coe_ne_zero.mpr hl')]
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l l' : NNReal
    hl' : Ne l' 0
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s l
    hf : HasConstantSpeedOnWith f (Set.image φ s) l'
    x : Real
    xs : Membership.mem s x
    y : Real
    ys : Membership.mem s y
    ⊢ Eq (HMul.hMul (HSub.hSub (φ y) (φ x)) ↑l') (HMul.hMul (HSub.hSub y x) ↑l)
  -/
  rw [hasConstantSpeedOnWith_iff_variationOnFromTo_eq] at hf
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l l' : NNReal
    hl' : Ne l' 0
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s l
    hf : And (LocallyBoundedVariationOn f (Set.image φ s)) (∀ ⦃x : Real⦄, Membersh …
    x : Real
    xs : Membership.mem s x
    y : Real
    ys : Membership.mem s y
    ⊢ Eq (HMul.hMul (HSub.hSub (φ y) (φ x)) ↑l') (HMul.hMul (HSub.hSub y x) ↑l)
  -/
  rw [hasConstantSpeedOnWith_iff_variationOnFromTo_eq] at hfφ
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    l l' : NNReal
    hl' : Ne l' 0
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : And (LocallyBoundedVariationOn (Function.comp f φ) s) (∀ ⦃x : Real⦄, Mem …
    hf : And (LocallyBoundedVariationOn f (Set.image φ s)) (∀ ⦃x : Real⦄, Membersh …
    x : Real
    xs : Membership.mem s x
    y : Real
    ys : Membership.mem s y
    ⊢ Eq (HMul.hMul (HSub.hSub (φ y) (φ x)) ↑l') (HMul.hMul (HSub.hSub y x) ↑l)
  -/
  symm
  calc
    (y - x) * l = l * (y - x) := by rw [mul_comm]
    _ = variationOnFromTo (f ∘ φ) s x y := (hfφ.2 xs ys).symm
    _ = variationOnFromTo f (φ '' s) (φ x) (φ y) :=
      (variationOnFromTo.comp_eq_of_monotoneOn f φ φm xs ys)
    _ = l' * (φ y - φ x) := (hf.2 ⟨x, xs, rfl⟩ ⟨y, ys, rfl⟩)
    _ = (φ y - φ x) * l' := by rw [mul_comm]


/-- `f` has unit speed on `s` if it is linearly parameterized by `l = 1` on `s`. -/
def HasUnitSpeedOn (f : ℝ → E) (s : Set ℝ) :=
  HasConstantSpeedOnWith f s 1


theorem HasUnitSpeedOn.union {t : Set ℝ} {x : ℝ} (hfs : HasUnitSpeedOn f s)
    (hft : HasUnitSpeedOn f t) (hs : IsGreatest s x) (ht : IsLeast t x) :
    HasUnitSpeedOn f (s ∪ t) :=
  HasConstantSpeedOnWith.union hfs hft hs ht


theorem HasUnitSpeedOn.Icc_Icc {x y z : ℝ} (hfs : HasUnitSpeedOn f (Icc x y))
    (hft : HasUnitSpeedOn f (Icc y z)) : HasUnitSpeedOn f (Icc x z) :=
  HasConstantSpeedOnWith.Icc_Icc hfs hft


/-- If both `f` and `f ∘ φ` have unit speed (on `t` and `s` respectively) and `φ`
monotonically maps `s` onto `t`, then `φ` is just a translation (on `s`).
-/
theorem unique_unit_speed {φ : ℝ → ℝ} (φm : MonotoneOn φ s) (hfφ : HasUnitSpeedOn (f ∘ φ) s)
    (hf : HasUnitSpeedOn f (φ '' s)) ⦃x : ℝ⦄ (xs : x ∈ s) : EqOn φ (fun y => y - x + φ x) s := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasUnitSpeedOn (Function.comp f φ) s
    hf : HasUnitSpeedOn f (Set.image φ s)
    x : Real
    xs : Membership.mem s x
    ⊢ Set.EqOn φ (fun y => HAdd.hAdd (HSub.hSub y x) (φ x)) s
  -/
  dsimp only [HasUnitSpeedOn] at hf hfφ
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s 1
    hf : HasConstantSpeedOnWith f (Set.image φ s) 1
    x : Real
    xs : Membership.mem s x
    ⊢ Set.EqOn φ (fun y => HAdd.hAdd (HSub.hSub y x) (φ x)) s
  -/
  convert HasConstantSpeedOnWith.ratio one_ne_zero φm hfφ hf xs using 3
  /-
    case h.e'_4.h.h.e'_5
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s : Set Real
    φ : Real → Real
    φm : MonotoneOn φ s
    hfφ : HasConstantSpeedOnWith (Function.comp f φ) s 1
    hf : HasConstantSpeedOnWith f (Set.image φ s) 1
    x : Real
    xs : Membership.mem s x
    x✝ : Real
    ⊢ Eq (HSub.hSub x✝ x) (HMul.hMul (HDiv.hDiv ↑1 ↑1) (HSub.hSub x✝ x))
  -/
  norm_num
  /-
    🎉 no goals
  -/


/-- If both `f` and `f ∘ φ` have unit speed (on `Icc 0 t` and `Icc 0 s` respectively)
and `φ` monotonically maps `Icc 0 s` onto `Icc 0 t`, then `φ` is the identity on `Icc 0 s`
-/
theorem unique_unit_speed_on_Icc_zero {s t : ℝ} (hs : 0 ≤ s) (ht : 0 ≤ t) {φ : ℝ → ℝ}
    (φm : MonotoneOn φ <| Icc 0 s) (φst : φ '' Icc 0 s = Icc 0 t)
    (hfφ : HasUnitSpeedOn (f ∘ φ) (Icc 0 s)) (hf : HasUnitSpeedOn f (Icc 0 t)) :
    EqOn φ id (Icc 0 s) := by
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s t : Real
    hs : LE.le 0 s
    ht : LE.le 0 t
    φ : Real → Real
    φm : MonotoneOn φ (Set.Icc 0 s)
    φst : Eq (Set.image φ (Set.Icc 0 s)) (Set.Icc 0 t)
    hfφ : HasUnitSpeedOn (Function.comp f φ) (Set.Icc 0 s)
    hf : HasUnitSpeedOn f (Set.Icc 0 t)
    ⊢ Set.EqOn φ id (Set.Icc 0 s)
  -/
  rw [← φst] at hf
  /-
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s t : Real
    hs : LE.le 0 s
    ht : LE.le 0 t
    φ : Real → Real
    φm : MonotoneOn φ (Set.Icc 0 s)
    φst : Eq (Set.image φ (Set.Icc 0 s)) (Set.Icc 0 t)
    hfφ : HasUnitSpeedOn (Function.comp f φ) (Set.Icc 0 s)
    hf : HasUnitSpeedOn f (Set.image φ (Set.Icc 0 s))
    ⊢ Set.EqOn φ id (Set.Icc 0 s)
  -/
  convert unique_unit_speed φm hfφ hf ⟨le_rfl, hs⟩ using 1
  have : φ 0 = 0 := by
    have hm : 0 ∈ φ '' Icc 0 s := by simp only [φst, ht, mem_Icc, le_refl, and_self]
    obtain ⟨x, xs, hx⟩ := hm
    apply le_antisymm ((φm ⟨le_rfl, hs⟩ xs xs.1).trans_eq hx) _
    have := φst ▸ mapsTo_image φ (Icc 0 s)
    exact (mem_Icc.mp (@this 0 (by rw [mem_Icc]; exact ⟨le_rfl, hs⟩))).1
  /-
    case h.e'_4
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s t : Real
    hs : LE.le 0 s
    ht : LE.le 0 t
    φ : Real → Real
    φm : MonotoneOn φ (Set.Icc 0 s)
    φst : Eq (Set.image φ (Set.Icc 0 s)) (Set.Icc 0 t)
    hfφ : HasUnitSpeedOn (Function.comp f φ) (Set.Icc 0 s)
    hf : HasUnitSpeedOn f (Set.image φ (Set.Icc 0 s))
    this : Eq (φ 0) 0
    ⊢ Eq id fun y => HAdd.hAdd (HSub.hSub y 0) (φ 0)
  -/
  simp only [tsub_zero, this, add_zero]
  /-
    case h.e'_4
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : Real → E
    s t : Real
    hs : LE.le 0 s
    ht : LE.le 0 t
    φ : Real → Real
    φm : MonotoneOn φ (Set.Icc 0 s)
    φst : Eq (Set.image φ (Set.Icc 0 s)) (Set.Icc 0 t)
    hfφ : HasUnitSpeedOn (Function.comp f φ) (Set.Icc 0 s)
    hf : HasUnitSpeedOn f (Set.image φ (Set.Icc 0 s))
    this : Eq (φ 0) 0
    ⊢ Eq id fun y => y
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The natural parameterization of `f` on `s`, which, if `f` has locally bounded variation on `s`,
* has unit speed on `s` (by `has_unit_speed_naturalParameterization`).
* composed with `variationOnFromTo f s a`, is at distance zero from `f`
  (by `edist_naturalParameterization_eq_zero`).
-/
noncomputable def naturalParameterization (f : α → E) (s : Set α) (a : α) : ℝ → E :=
  f ∘ @Function.invFunOn _ _ ⟨a⟩ (variationOnFromTo f s a) s


theorem edist_naturalParameterization_eq_zero {f : α → E} {s : Set α}
    (hf : LocallyBoundedVariationOn f s) {a : α} (as : a ∈ s) {b : α} (bs : b ∈ s) :
    edist (naturalParameterization f s a (variationOnFromTo f s a b)) (f b) = 0 := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    ⊢ Eq (EDist.edist (naturalParameterization f s a (variationOnFromTo f s a b))  …
  -/
  dsimp only [naturalParameterization]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    ⊢ Eq (EDist.edist (Function.comp f (Function.invFunOn (variationOnFromTo f s a …
  -/
  haveI : Nonempty α := ⟨a⟩
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    this : Nonempty α
    ⊢ Eq (EDist.edist (Function.comp f (Function.invFunOn (variationOnFromTo f s a …
  -/
  obtain ⟨cs, hc⟩ := Function.invFunOn_pos (b := variationOnFromTo f s a b) ⟨b, bs, rfl⟩
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    this : Nonempty α
    cs : Membership.mem s (Function.invFunOn (variationOnFromTo f s a) s (variatio …
    hc : Eq (variationOnFromTo f s a (Function.invFunOn (variationOnFromTo f s a)  …
    ⊢ Eq (EDist.edist (Function.comp f (Function.invFunOn (variationOnFromTo f s a …
  -/
  rw [variationOnFromTo.eq_left_iff hf as cs bs] at hc
  /-
    case intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    this : Nonempty α
    cs : Membership.mem s (Function.invFunOn (variationOnFromTo f s a) s (variatio …
    hc : Eq (variationOnFromTo f s (Function.invFunOn (variationOnFromTo f s a) s  …
    ⊢ Eq (EDist.edist (Function.comp f (Function.invFunOn (variationOnFromTo f s a …
  -/
  apply variationOnFromTo.edist_zero_of_eq_zero hf cs bs hc
  /-
    🎉 no goals
  -/


theorem has_unit_speed_naturalParameterization (f : α → E) {s : Set α}
    (hf : LocallyBoundedVariationOn f s) {a : α} (as : a ∈ s) :
    HasUnitSpeedOn (naturalParameterization f s a) (variationOnFromTo f s a '' s) := by
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    ⊢ HasUnitSpeedOn (naturalParameterization f s a) (Set.image (variationOnFromTo …
  -/
  dsimp only [HasUnitSpeedOn]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    ⊢ HasConstantSpeedOnWith (naturalParameterization f s a) (Set.image (variation …
  -/
  rw [hasConstantSpeedOnWith_iff_ordered]
  /-
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    ⊢ ∀ ⦃x : Real⦄, Membership.mem (Set.image (variationOnFromTo f s a) s) x → ∀ ⦃ …
  -/
  rintro _ ⟨b, bs, rfl⟩ _ ⟨c, cs, rfl⟩ h
  /-
    case intro.intro.intro.intro
    α : Type u_1
    inst✝¹ : LinearOrder α
    E : Type u_2
    inst✝ : PseudoEMetricSpace E
    f : α → E
    s : Set α
    hf : LocallyBoundedVariationOn f s
    a : α
    as : Membership.mem s a
    b : α
    bs : Membership.mem s b
    c : α
    cs : Membership.mem s c
    h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
    ⊢ Eq (eVariationOn (naturalParameterization f s a) (Inter.inter (Set.image (va …
  -/
  rcases le_total c b with (cb | bc)
  · rw [NNReal.coe_one, one_mul, le_antisymm h (variationOnFromTo.monotoneOn hf as cs bs cb),
      sub_self, ENNReal.ofReal_zero, Icc_self, eVariationOn.subsingleton]
    /-
      case intro.intro.intro.intro.inl.hs
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a : α
      as : Membership.mem s a
      b : α
      bs : Membership.mem s b
      c : α
      cs : Membership.mem s c
      h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
      cb : LE.le c b
      ⊢ (Inter.inter (Set.image (variationOnFromTo f s a) s) (Singleton.singleton (v …
    -/
    exact fun x hx y hy => hx.2.trans hy.2.symm
    /-
      🎉 no goals
    -/
  · rw [NNReal.coe_one, one_mul, sub_eq_add_neg, variationOnFromTo.eq_neg_swap, neg_neg, add_comm,
      variationOnFromTo.add hf bs as cs, ← variationOnFromTo.eq_neg_swap f]
    rw [←
      eVariationOn.comp_inter_Icc_eq_of_monotoneOn (naturalParameterization f s a) _
        (variationOnFromTo.monotoneOn hf as) bs cs]
    /-
      case intro.intro.intro.intro.inr
      α : Type u_1
      inst✝¹ : LinearOrder α
      E : Type u_2
      inst✝ : PseudoEMetricSpace E
      f : α → E
      s : Set α
      hf : LocallyBoundedVariationOn f s
      a : α
      as : Membership.mem s a
      b : α
      bs : Membership.mem s b
      c : α
      cs : Membership.mem s c
      h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
      bc : LE.le b c
      ⊢ Eq (eVariationOn (Function.comp (naturalParameterization f s a) (variationOn …
    -/
    rw [@eVariationOn.eq_of_edist_zero_on _ _ _ _ _ f]
      /-
        case intro.intro.intro.intro.inr
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        hf : LocallyBoundedVariationOn f s
        a : α
        as : Membership.mem s a
        b : α
        bs : Membership.mem s b
        c : α
        cs : Membership.mem s c
        h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
        bc : LE.le b c
        ⊢ Eq (eVariationOn f (Inter.inter s (Set.Icc b c))) (ENNReal.ofReal (variation …
      -/
    · rw [variationOnFromTo.eq_of_le _ _ bc, ENNReal.ofReal_toReal (hf b c bs cs)]
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.inr.h
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        hf : LocallyBoundedVariationOn f s
        a : α
        as : Membership.mem s a
        b : α
        bs : Membership.mem s b
        c : α
        cs : Membership.mem s c
        h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
        bc : LE.le b c
        ⊢ ∀ ⦃x : α⦄, Membership.mem (Inter.inter s (Set.Icc b c)) x → Eq (EDist.edist  …
      -/
    · rintro x ⟨xs, _, _⟩
      /-
        case intro.intro.intro.intro.inr.h.intro.intro
        α : Type u_1
        inst✝¹ : LinearOrder α
        E : Type u_2
        inst✝ : PseudoEMetricSpace E
        f : α → E
        s : Set α
        hf : LocallyBoundedVariationOn f s
        a : α
        as : Membership.mem s a
        b : α
        bs : Membership.mem s b
        c : α
        cs : Membership.mem s c
        h : LE.le (variationOnFromTo f s a b) (variationOnFromTo f s a c)
        bc : LE.le b c
        x : α
        xs : Membership.mem s x
        left✝ : LE.le b x
        right✝ : LE.le x c
        ⊢ Eq (EDist.edist (Function.comp (naturalParameterization f s a) (variationOnF …
      -/
      exact edist_naturalParameterization_eq_zero hf as xs
      /-
        🎉 no goals
      -/

