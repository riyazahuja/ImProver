lemma strictConcaveOn_rpow {p : ℝ} (hp₀ : 0 < p) (hp₁ : p < 1) :
    StrictConcaveOn ℝ≥0 univ fun x : ℝ≥0 ↦ x ^ p := by
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  have hp₀' : 0 < 1 / p := div_pos zero_lt_one hp₀
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  have hp₁' : 1 < 1 / p := by rw [one_lt_div hp₀]; exact hp₁
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    hp₁' : LT.lt 1 (HDiv.hDiv 1 p)
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  let f := NNReal.orderIsoRpow (1 / p) hp₀'
  have h₁ : StrictConvexOn ℝ≥0 univ f := by
    refine ⟨convex_univ, fun x _ y _ hxy a b ha hb hab => ?_⟩
    exact (strictConvexOn_rpow hp₁').2 x.2 y.2 (by simp [hxy]) ha hb (by simp; norm_cast)
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    hp₁' : LT.lt 1 (HDiv.hDiv 1 p)
    f : OrderIso NNReal NNReal := NNReal.orderIsoRpow (HDiv.hDiv 1 p) hp₀'
    h₁ : StrictConvexOn NNReal Set.univ ⇑f
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  have h₂ : ∀ x, f.symm x = x ^ p := by simp [f, NNReal.orderIsoRpow_symm_eq]
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    hp₁' : LT.lt 1 (HDiv.hDiv 1 p)
    f : OrderIso NNReal NNReal := NNReal.orderIsoRpow (HDiv.hDiv 1 p) hp₀'
    h₁ : StrictConvexOn NNReal Set.univ ⇑f
    h₂ : ∀ (x : NNReal), Eq (f.symm x) (HPow.hPow x p)
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  refine ⟨convex_univ, fun x mx y my hxy a b ha hb hab => ?_⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    hp₁' : LT.lt 1 (HDiv.hDiv 1 p)
    f : OrderIso NNReal NNReal := NNReal.orderIsoRpow (HDiv.hDiv 1 p) hp₀'
    h₁ : StrictConvexOn NNReal Set.univ ⇑f
    h₂ : ∀ (x : NNReal), Eq (f.symm x) (HPow.hPow x p)
    x : NNReal
    mx : Membership.mem Set.univ x
    y : NNReal
    my : Membership.mem Set.univ y
    hxy : Ne x y
    a b : NNReal
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  simp only [← h₂]
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    hp₀' : LT.lt 0 (HDiv.hDiv 1 p)
    hp₁' : LT.lt 1 (HDiv.hDiv 1 p)
    f : OrderIso NNReal NNReal := NNReal.orderIsoRpow (HDiv.hDiv 1 p) hp₀'
    h₁ : StrictConvexOn NNReal Set.univ ⇑f
    h₂ : ∀ (x : NNReal), Eq (f.symm x) (HPow.hPow x p)
    x : NNReal
    mx : Membership.mem Set.univ x
    y : NNReal
    my : Membership.mem Set.univ y
    hxy : Ne x y
    a b : NNReal
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a (f.symm x)) (HSMul.hSMul b (f.symm y))) (f.s …
  -/
  exact (f.strictConcaveOn_symm h₁).2 mx my hxy ha hb hab
  /-
    🎉 no goals
  -/


lemma concaveOn_rpow {p : ℝ} (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) :
    ConcaveOn ℝ≥0 univ fun x : ℝ≥0 ↦ x ^ p := by
  /-
    p : Real
    hp₀ : LE.le 0 p
    hp₁ : LE.le p 1
    ⊢ ConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  rcases eq_or_lt_of_le hp₀ with (rfl | hp₀)
    /-
      case inl
      hp₀ : LE.le 0 0
      hp₁ : LE.le 0 1
      ⊢ ConcaveOn NNReal Set.univ fun x => HPow.hPow x 0
    -/
  · simpa only [rpow_zero] using concaveOn_const (c := 1) convex_univ
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁ : LE.le p 1
    hp₀ : LT.lt 0 p
    ⊢ ConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  rcases eq_or_lt_of_le hp₁ with (rfl | hp₁)
    /-
      case inr.inl
      hp₀✝ : LE.le 0 1
      hp₁ : LE.le 1 1
      hp₀ : LT.lt 0 1
      ⊢ ConcaveOn NNReal Set.univ fun x => HPow.hPow x 1
    -/
  · simpa only [rpow_one] using concaveOn_id convex_univ
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁✝ : LE.le p 1
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ ConcaveOn NNReal Set.univ fun x => HPow.hPow x p
  -/
  exact (strictConcaveOn_rpow hp₀ hp₁).concaveOn
  /-
    🎉 no goals
  -/


lemma strictConcaveOn_sqrt : StrictConcaveOn ℝ≥0 univ NNReal.sqrt := by
  have : NNReal.sqrt = fun x : ℝ≥0 ↦ x ^ (1 / (2 : ℝ)) := by
    ext x; exact mod_cast NNReal.sqrt_eq_rpow x
  /-
    this : Eq ⇑NNReal.sqrt fun x => HPow.hPow x (1 / 2)
    ⊢ StrictConcaveOn NNReal Set.univ ⇑NNReal.sqrt
  -/
  rw [this]
  /-
    this : Eq ⇑NNReal.sqrt fun x => HPow.hPow x (1 / 2)
    ⊢ StrictConcaveOn NNReal Set.univ fun x => HPow.hPow x (1 / 2)
  -/
  exact strictConcaveOn_rpow (by positivity) (by linarith)
  /-
    🎉 no goals
  -/


lemma strictConcaveOn_rpow {p : ℝ} (hp₀ : 0 < p) (hp₁ : p < 1) :
    StrictConcaveOn ℝ (Set.Ici 0) fun x : ℝ ↦ x ^ p := by
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ StrictConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  refine ⟨convex_Ici _, fun x hx y hy hxy a b ha hb hab => ?_⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  let x' : ℝ≥0 := ⟨x, hx⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : NNReal := ⟨x, hx⟩
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  let y' : ℝ≥0 := ⟨y, hy⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : NNReal := ⟨x, hx⟩
    y' : NNReal := ⟨y, hy⟩
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  let a' : ℝ≥0 := ⟨a, ha.le⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : NNReal := ⟨x, hx⟩
    y' : NNReal := ⟨y, hy⟩
    a' : NNReal := ⟨a, ⋯⟩
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  let b' : ℝ≥0 := ⟨b, hb.le⟩
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : NNReal := ⟨x, hx⟩
    y' : NNReal := ⟨y, hy⟩
    a' : NNReal := ⟨a, ⋯⟩
    b' : NNReal := ⟨b, ⋯⟩
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  have hxy' : x' ≠ y' := Subtype.coe_ne_coe.1 hxy
  /-
    p : Real
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    x : Real
    hx : Membership.mem (Set.Ici 0) x
    y : Real
    hy : Membership.mem (Set.Ici 0) y
    hxy : Ne x y
    a b : Real
    ha : LT.lt 0 a
    hb : LT.lt 0 b
    hab : Eq (HAdd.hAdd a b) 1
    x' : NNReal := ⟨x, hx⟩
    y' : NNReal := ⟨y, hy⟩
    a' : NNReal := ⟨a, ⋯⟩
    b' : NNReal := ⟨b, ⋯⟩
    hxy' : Ne x' y'
    ⊢ LT.lt (HAdd.hAdd (HSMul.hSMul a ((fun x => HPow.hPow x p) x)) (HSMul.hSMul b …
  -/
  have hab' : a' + b' = 1 := by ext; simp [a', b', hab]
  exact_mod_cast (NNReal.strictConcaveOn_rpow hp₀ hp₁).2 (Set.mem_univ x') (Set.mem_univ y')
    hxy' (mod_cast ha) (mod_cast hb) hab'


lemma concaveOn_rpow {p : ℝ} (hp₀ : 0 ≤ p) (hp₁ : p ≤ 1) :
    ConcaveOn ℝ (Set.Ici 0) fun x : ℝ ↦ x ^ p := by
  /-
    p : Real
    hp₀ : LE.le 0 p
    hp₁ : LE.le p 1
    ⊢ ConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  rcases eq_or_lt_of_le hp₀ with (rfl | hp₀)
    /-
      case inl
      hp₀ : LE.le 0 0
      hp₁ : LE.le 0 1
      ⊢ ConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x 0
    -/
  · simpa only [rpow_zero] using concaveOn_const (c := 1) (convex_Ici _)
    /-
      🎉 no goals
    -/
  /-
    case inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁ : LE.le p 1
    hp₀ : LT.lt 0 p
    ⊢ ConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  rcases eq_or_lt_of_le hp₁ with (rfl | hp₁)
    /-
      case inr.inl
      hp₀✝ : LE.le 0 1
      hp₁ : LE.le 1 1
      hp₀ : LT.lt 0 1
      ⊢ ConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x 1
    -/
  · simpa only [rpow_one] using concaveOn_id (convex_Ici _)
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    p : Real
    hp₀✝ : LE.le 0 p
    hp₁✝ : LE.le p 1
    hp₀ : LT.lt 0 p
    hp₁ : LT.lt p 1
    ⊢ ConcaveOn Real (Set.Ici 0) fun x => HPow.hPow x p
  -/
  exact (strictConcaveOn_rpow hp₀ hp₁).concaveOn
  /-
    🎉 no goals
  -/


lemma strictConcaveOn_sqrt : StrictConcaveOn ℝ (Set.Ici 0) (√· : ℝ → ℝ) := by
  /-
    ⊢ StrictConcaveOn Real (Set.Ici 0) fun x => x.sqrt
  -/
  rw [funext Real.sqrt_eq_rpow]
  /-
    ⊢ StrictConcaveOn Real (Set.Ici 0) fun x => (fun x => HPow.hPow x (1 / 2)) x
  -/
  exact strictConcaveOn_rpow (by positivity) (by linarith)
  /-
    🎉 no goals
  -/


