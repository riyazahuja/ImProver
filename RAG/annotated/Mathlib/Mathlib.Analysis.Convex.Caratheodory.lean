/-- If `x` is in the convex hull of some finset `t` whose elements are not affine-independent,
then it is in the convex hull of a strict subset of `t`. -/
theorem mem_convexHull_erase [DecidableEq E] {t : Finset E} (h : ¬AffineIndependent 𝕜 ((↑) : t → E))
    {x : E} (m : x ∈ convexHull 𝕜 (↑t : Set E)) :
    ∃ y : (↑t : Set E), x ∈ convexHull 𝕜 (↑(t.erase y) : Set E) := by
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    h : Not (AffineIndependent 𝕜 Subtype.val)
    x : E
    m : Membership.mem ((convexHull 𝕜) ↑t) x
    ⊢ Exists fun y => Membership.mem ((convexHull 𝕜) ↑(t.erase ↑y)) x
  -/
  simp only [Finset.convexHull_eq, mem_setOf_eq] at m ⊢
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    h : Not (AffineIndependent 𝕜 Subtype.val)
    x : E
    m : Exists fun w => And (∀ (y : E), Membership.mem t y → LE.le 0 (w y)) (And ( …
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  obtain ⟨f, fpos, fsum, rfl⟩ := m
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    h : Not (AffineIndependent 𝕜 Subtype.val)
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  obtain ⟨g, gcombo, gsum, gpos⟩ := exists_nontrivial_relation_sum_zero_of_not_affine_ind h
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    h : Not (AffineIndependent 𝕜 Subtype.val)
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun x => And (Membership.mem t x) (Ne (g x) 0)
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  replace gpos := exists_pos_of_sum_zero_of_exists_nonzero g gsum gpos
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    h : Not (AffineIndependent 𝕜 Subtype.val)
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  clear h
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  let s := {z ∈ t | 0 < g z}
  obtain ⟨i₀, mem, w⟩ : ∃ i₀ ∈ s, ∀ i ∈ s, f i₀ / g i₀ ≤ f i / g i := by
    apply s.exists_min_image fun z => f z / g z
    obtain ⟨x, hx, hgx⟩ : ∃ x ∈ t, 0 < g x := gpos
    exact ⟨x, mem_filter.mpr ⟨hx, hgx⟩⟩
  have hg : 0 < g i₀ := by
    rw [mem_filter] at mem
    exact mem.2
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
    i₀ : E
    mem : Membership.mem s i₀
    w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
    hg : LT.lt 0 (g i₀)
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  have hi₀ : i₀ ∈ t := filter_subset _ _ mem
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
    i₀ : E
    mem : Membership.mem s i₀
    w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
    hg : LT.lt 0 (g i₀)
    hi₀ : Membership.mem t i₀
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  let k : E → 𝕜 := fun z => f z - f i₀ / g i₀ * g z
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
    i₀ : E
    mem : Membership.mem s i₀
    w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
    hg : LT.lt 0 (g i₀)
    hi₀ : Membership.mem t i₀
    k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  have hk : k i₀ = 0 := by field_simp [k, ne_of_gt hg]
  have ksum : ∑ e ∈ t.erase i₀, k e = 1 := by
    calc
      ∑ e ∈ t.erase i₀, k e = ∑ e ∈ t, k e := by
        conv_rhs => rw [← insert_erase hi₀, sum_insert (not_mem_erase i₀ t), hk, zero_add]
      _ = ∑ e ∈ t, (f e - f i₀ / g i₀ * g e) := rfl
      _ = 1 := by rw [sum_sub_distrib, fsum, ← mul_sum, gsum, mul_zero, sub_zero]
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝³ : LinearOrderedField 𝕜
    inst✝² : AddCommGroup E
    inst✝¹ : Module 𝕜 E
    inst✝ : DecidableEq E
    t : Finset E
    f : E → 𝕜
    fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
    fsum : Eq (t.sum fun y => f y) 1
    g : E → 𝕜
    gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
    gsum : Eq (t.sum fun e => g e) 0
    gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
    s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
    i₀ : E
    mem : Membership.mem s i₀
    w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
    hg : LT.lt 0 (g i₀)
    hi₀ : Membership.mem t i₀
    k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
    hk : Eq (k i₀) 0
    ksum : Eq ((t.erase i₀).sum fun e => k e) 1
    ⊢ Exists fun y => Exists fun w => And (∀ (y_1 : E), Membership.mem (t.erase ↑y …
  -/
  refine ⟨⟨i₀, hi₀⟩, k, ?_, by convert ksum, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : DecidableEq E
      t : Finset E
      f : E → 𝕜
      fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
      fsum : Eq (t.sum fun y => f y) 1
      g : E → 𝕜
      gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
      gsum : Eq (t.sum fun e => g e) 0
      gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
      s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
      i₀ : E
      mem : Membership.mem s i₀
      w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
      hg : LT.lt 0 (g i₀)
      hi₀ : Membership.mem t i₀
      k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
      hk : Eq (k i₀) 0
      ksum : Eq ((t.erase i₀).sum fun e => k e) 1
      ⊢ ∀ (y : E), Membership.mem (t.erase ↑⟨i₀, hi₀⟩) y → LE.le 0 (k y)
    -/
  · simp only [k, and_imp, sub_nonneg, mem_erase, Ne, Subtype.coe_mk]
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : DecidableEq E
      t : Finset E
      f : E → 𝕜
      fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
      fsum : Eq (t.sum fun y => f y) 1
      g : E → 𝕜
      gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
      gsum : Eq (t.sum fun e => g e) 0
      gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
      s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
      i₀ : E
      mem : Membership.mem s i₀
      w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
      hg : LT.lt 0 (g i₀)
      hi₀ : Membership.mem t i₀
      k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
      hk : Eq (k i₀) 0
      ksum : Eq ((t.erase i₀).sum fun e => k e) 1
      ⊢ ∀ (y : E), Not (Eq y i₀) → Membership.mem t y → LE.le (HMul.hMul (HDiv.hDiv  …
    -/
    intro e _ het
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : DecidableEq E
      t : Finset E
      f : E → 𝕜
      fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
      fsum : Eq (t.sum fun y => f y) 1
      g : E → 𝕜
      gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
      gsum : Eq (t.sum fun e => g e) 0
      gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
      s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
      i₀ : E
      mem : Membership.mem s i₀
      w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
      hg : LT.lt 0 (g i₀)
      hi₀ : Membership.mem t i₀
      k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
      hk : Eq (k i₀) 0
      ksum : Eq ((t.erase i₀).sum fun e => k e) 1
      e : E
      a✝ : Not (Eq e i₀)
      het : Membership.mem t e
      ⊢ LE.le (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g e)) (f e)
    -/
    by_cases hes : e ∈ s
    · have hge : 0 < g e := by
        rw [mem_filter] at hes
        exact hes.2
      /-
        case pos
        𝕜 : Type u_1
        E : Type u
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : DecidableEq E
        t : Finset E
        f : E → 𝕜
        fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
        fsum : Eq (t.sum fun y => f y) 1
        g : E → 𝕜
        gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
        gsum : Eq (t.sum fun e => g e) 0
        gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
        s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
        i₀ : E
        mem : Membership.mem s i₀
        w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
        hg : LT.lt 0 (g i₀)
        hi₀ : Membership.mem t i₀
        k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
        hk : Eq (k i₀) 0
        ksum : Eq ((t.erase i₀).sum fun e => k e) 1
        e : E
        a✝ : Not (Eq e i₀)
        het : Membership.mem t e
        hes : Membership.mem s e
        hge : LT.lt 0 (g e)
        ⊢ LE.le (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g e)) (f e)
      -/
      rw [← le_div_iff₀ hge]
      /-
        case pos
        𝕜 : Type u_1
        E : Type u
        inst✝³ : LinearOrderedField 𝕜
        inst✝² : AddCommGroup E
        inst✝¹ : Module 𝕜 E
        inst✝ : DecidableEq E
        t : Finset E
        f : E → 𝕜
        fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
        fsum : Eq (t.sum fun y => f y) 1
        g : E → 𝕜
        gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
        gsum : Eq (t.sum fun e => g e) 0
        gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
        s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
        i₀ : E
        mem : Membership.mem s i₀
        w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
        hg : LT.lt 0 (g i₀)
        hi₀ : Membership.mem t i₀
        k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
        hk : Eq (k i₀) 0
        ksum : Eq ((t.erase i₀).sum fun e => k e) 1
        e : E
        a✝ : Not (Eq e i₀)
        het : Membership.mem t e
        hes : Membership.mem s e
        hge : LT.lt 0 (g e)
        ⊢ LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv (f e) (g e))
      -/
      exact w _ hes
      /-
        🎉 no goals
      -/
    · calc
        _ ≤ 0 := by
          apply mul_nonpos_of_nonneg_of_nonpos
          · apply div_nonneg (fpos i₀ (mem_of_subset (filter_subset _ t) mem)) (le_of_lt hg)
          · simpa only [s, mem_filter, het, true_and, not_lt] using hes
        _ ≤ f e := fpos e het
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      E : Type u
      inst✝³ : LinearOrderedField 𝕜
      inst✝² : AddCommGroup E
      inst✝¹ : Module 𝕜 E
      inst✝ : DecidableEq E
      t : Finset E
      f : E → 𝕜
      fpos : ∀ (y : E), Membership.mem t y → LE.le 0 (f y)
      fsum : Eq (t.sum fun y => f y) 1
      g : E → 𝕜
      gcombo : Eq (t.sum fun e => HSMul.hSMul (g e) e) 0
      gsum : Eq (t.sum fun e => g e) 0
      gpos : Exists fun i => And (Membership.mem t i) (LT.lt 0 (g i))
      s : Finset E := Finset.filter (fun z => LT.lt 0 (g z)) t
      i₀ : E
      mem : Membership.mem s i₀
      w : ∀ (i : E), Membership.mem s i → LE.le (HDiv.hDiv (f i₀) (g i₀)) (HDiv.hDiv …
      hg : LT.lt 0 (g i₀)
      hi₀ : Membership.mem t i₀
      k : E → 𝕜 := fun z => HSub.hSub (f z) (HMul.hMul (HDiv.hDiv (f i₀) (g i₀)) (g  …
      hk : Eq (k i₀) 0
      ksum : Eq ((t.erase i₀).sum fun e => k e) 1
      ⊢ Eq ((t.erase ↑⟨i₀, hi₀⟩).centerMass k id) (t.centerMass f id)
    -/
  · rw [Subtype.coe_mk, centerMass_eq_of_sum_1 _ id ksum]
    calc
      ∑ e ∈ t.erase i₀, k e • e = ∑ e ∈ t, k e • e := sum_erase _ (by rw [hk, zero_smul])
      _ = ∑ e ∈ t, (f e - f i₀ / g i₀ * g e) • e := rfl
      _ = t.centerMass f id := by
        simp only [sub_smul, mul_smul, sum_sub_distrib, ← smul_sum, gcombo, smul_zero, sub_zero,
          centerMass, fsum, inv_one, one_smul, id]


/-- Given a point `x` in the convex hull of a set `s`, this is a finite subset of `s` of minimum
cardinality, whose convex hull contains `x`. -/
noncomputable def minCardFinsetOfMemConvexHull (hx : x ∈ convexHull 𝕜 s) : Finset E :=
  Function.argminOn Finset.card Nat.lt_wfRel.2 { t | ↑t ⊆ s ∧ x ∈ convexHull 𝕜 (t : Set E) } <| by
    /-
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      hx : Membership.mem ((convexHull 𝕜) s) x
      ⊢ (setOf fun t => And (HasSubset.Subset (↑t) s) (Membership.mem ((convexHull 𝕜 …
    -/
    simpa only [convexHull_eq_union_convexHull_finite_subsets s, exists_prop, mem_iUnion] using hx
    /-
      🎉 no goals
    -/


theorem minCardFinsetOfMemConvexHull_subseteq : ↑(minCardFinsetOfMemConvexHull hx) ⊆ s :=
  (Function.argminOn_mem _ _ { t : Finset E | ↑t ⊆ s ∧ x ∈ convexHull 𝕜 (t : Set E) } _).1


theorem mem_minCardFinsetOfMemConvexHull :
    x ∈ convexHull 𝕜 (minCardFinsetOfMemConvexHull hx : Set E) :=
  (Function.argminOn_mem _ _ { t : Finset E | ↑t ⊆ s ∧ x ∈ convexHull 𝕜 (t : Set E) } _).2


theorem minCardFinsetOfMemConvexHull_nonempty : (minCardFinsetOfMemConvexHull hx).Nonempty := by
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    ⊢ (Caratheodory.minCardFinsetOfMemConvexHull hx).Nonempty
  -/
  rw [← Finset.coe_nonempty, ← @convexHull_nonempty_iff 𝕜]
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    ⊢ ((convexHull 𝕜) ↑(Caratheodory.minCardFinsetOfMemConvexHull hx)).Nonempty
  -/
  exact ⟨x, mem_minCardFinsetOfMemConvexHull hx⟩
  /-
    🎉 no goals
  -/


theorem minCardFinsetOfMemConvexHull_card_le_card {t : Finset E} (ht₁ : ↑t ⊆ s)
    (ht₂ : x ∈ convexHull 𝕜 (t : Set E)) : #(minCardFinsetOfMemConvexHull hx) ≤ #t :=
                                 /-
                                   𝕜 : Type u_1
                                   E : Type u
                                   inst✝² : LinearOrderedField 𝕜
                                   inst✝¹ : AddCommGroup E
                                   inst✝ : Module 𝕜 E
                                   s : Set E
                                   x : E
                                   hx : Membership.mem ((convexHull 𝕜) s) x
                                   t : Finset E
                                   ht₁ : HasSubset.Subset (↑t) s
                                   ht₂ : Membership.mem ((convexHull 𝕜) ↑t) x
                                   ⊢ Membership.mem (setOf fun t => And (HasSubset.Subset (↑t) s) (Membership.mem …
                                 -/
  Function.argminOn_le _ _ _ (by exact ⟨ht₁, ht₂⟩)
                                 /-
                                   🎉 no goals
                                 -/


theorem affineIndependent_minCardFinsetOfMemConvexHull :
    AffineIndependent 𝕜 ((↑) : minCardFinsetOfMemConvexHull hx → E) := by
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    ⊢ AffineIndependent 𝕜 Subtype.val
  -/
  let k := #(minCardFinsetOfMemConvexHull hx) - 1
  have hk : #(minCardFinsetOfMemConvexHull hx) = k + 1 :=
    (Nat.succ_pred_eq_of_pos (Finset.card_pos.mpr (minCardFinsetOfMemConvexHull_nonempty hx))).symm
  classical
  by_contra h
  obtain ⟨p, hp⟩ := mem_convexHull_erase h (mem_minCardFinsetOfMemConvexHull hx)
  have contra := minCardFinsetOfMemConvexHull_card_le_card hx (Set.Subset.trans
    (Finset.erase_subset (p : E) (minCardFinsetOfMemConvexHull hx))
    (minCardFinsetOfMemConvexHull_subseteq hx)) hp
  rw [← not_lt] at contra
  apply contra
  rw [card_erase_of_mem p.2, hk]
  exact lt_add_one _


/-- **Carathéodory's convexity theorem** -/
theorem convexHull_eq_union : convexHull 𝕜 s =
    ⋃ (t : Finset E) (_ : ↑t ⊆ s) (_ : AffineIndependent 𝕜 ((↑) : t → E)), convexHull 𝕜 ↑t := by
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    ⊢ Eq ((convexHull 𝕜) s) (Set.iUnion fun t => Set.iUnion fun x => Set.iUnion fu …
  -/
  apply Set.Subset.antisymm
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ HasSubset.Subset ((convexHull 𝕜) s) (Set.iUnion fun t => Set.iUnion fun x => …
    -/
  · intro x hx
    /-
      case h₁
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      hx : Membership.mem ((convexHull 𝕜) s) x
      ⊢ Membership.mem (Set.iUnion fun t => Set.iUnion fun x => Set.iUnion fun x =>  …
    -/
    simp only [exists_prop, Set.mem_iUnion]
    exact ⟨Caratheodory.minCardFinsetOfMemConvexHull hx,
      Caratheodory.minCardFinsetOfMemConvexHull_subseteq hx,
      Caratheodory.affineIndependent_minCardFinsetOfMemConvexHull hx,
      Caratheodory.mem_minCardFinsetOfMemConvexHull hx⟩
    /-
      case h₂
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      ⊢ HasSubset.Subset (Set.iUnion fun t => Set.iUnion fun x => Set.iUnion fun x = …
    -/
  · iterate 3 convert Set.iUnion_subset _; intro
    /-
      case h₂.convert_5.convert_5.convert_5
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      i✝² : Finset E
      i✝¹ : HasSubset.Subset (↑i✝²) s
      i✝ : AffineIndependent 𝕜 Subtype.val
      ⊢ HasSubset.Subset ((convexHull 𝕜) ↑i✝²) ((convexHull 𝕜) s)
    -/
    exact convexHull_mono ‹_›
    /-
      🎉 no goals
    -/


/-- A more explicit version of `convexHull_eq_union`. -/
theorem eq_pos_convex_span_of_mem_convexHull {x : E} (hx : x ∈ convexHull 𝕜 s) :
    ∃ (ι : Sort (u + 1)) (_ : Fintype ι),
      ∃ (z : ι → E) (w : ι → 𝕜), Set.range z ⊆ s ∧ AffineIndependent 𝕜 z ∧ (∀ i, 0 < w i) ∧
        ∑ i, w i = 1 ∧ ∑ i, w i • z i = x := by
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem ((convexHull 𝕜) s) x
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  rw [convexHull_eq_union] at hx
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Membership.mem (Set.iUnion fun t => Set.iUnion fun x => Set.iUnion fun x  …
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  simp only [exists_prop, Set.mem_iUnion] at hx
  /-
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    hx : Exists fun i => And (HasSubset.Subset (↑i) s) (And (AffineIndependent 𝕜 S …
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  obtain ⟨t, ht₁, ht₂, ht₃⟩ := hx
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    t : Finset E
    ht₁ : HasSubset.Subset (↑t) s
    ht₂ : AffineIndependent 𝕜 Subtype.val
    ht₃ : Membership.mem ((convexHull 𝕜) ↑t) x
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  simp only [t.convexHull_eq, exists_prop, Set.mem_setOf_eq] at ht₃
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    t : Finset E
    ht₁ : HasSubset.Subset (↑t) s
    ht₂ : AffineIndependent 𝕜 Subtype.val
    ht₃ : Exists fun w => And (∀ (y : E), Membership.mem t y → LE.le 0 (w y)) (And …
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  obtain ⟨w, hw₁, hw₂, hw₃⟩ := ht₃
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    t : Finset E
    ht₁ : HasSubset.Subset (↑t) s
    ht₂ : AffineIndependent 𝕜 Subtype.val
    w : E → 𝕜
    hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
    hw₂ : Eq (t.sum fun y => w y) 1
    hw₃ : Eq (t.centerMass w id) x
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  let t' := {i ∈ t | w i ≠ 0}
  /-
    case intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u
    inst✝² : LinearOrderedField 𝕜
    inst✝¹ : AddCommGroup E
    inst✝ : Module 𝕜 E
    s : Set E
    x : E
    t : Finset E
    ht₁ : HasSubset.Subset (↑t) s
    ht₂ : AffineIndependent 𝕜 Subtype.val
    w : E → 𝕜
    hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
    hw₂ : Eq (t.sum fun y => w y) 1
    hw₃ : Eq (t.centerMass w id) x
    t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
    ⊢ Exists fun ι => Exists fun x_1 => Exists fun z => Exists fun w => And (HasSu …
  -/
  refine ⟨t', t'.fintypeCoeSort, ((↑) : t' → E), w ∘ ((↑) : t' → E), ?_, ?_, ?_, ?_, ?_⟩
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ HasSubset.Subset (Set.range Subtype.val) s
    -/
  · rw [Subtype.range_coe_subtype]
    /-
      case intro.intro.intro.intro.intro.intro.refine_1
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ HasSubset.Subset (setOf fun x => Membership.mem t' x) s
    -/
    exact Subset.trans (Finset.filter_subset _ t) ht₁
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_2
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ AffineIndependent 𝕜 Subtype.val
    -/
  · exact ht₂.comp_embedding ⟨_, inclusion_injective (Finset.filter_subset (fun i => w i ≠ 0) t)⟩
    /-
      🎉 no goals
    -/
  · exact fun i =>
      (hw₁ _ (Finset.mem_filter.mp i.2).1).lt_of_ne (Finset.mem_filter.mp i.property).2.symm
    /-
      case intro.intro.intro.intro.intro.intro.refine_4
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ Eq (Finset.univ.sum fun i => Function.comp w Subtype.val i) 1
    -/
  · erw [Finset.sum_attach, Finset.sum_filter_ne_zero, hw₂]
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.intro.intro.intro.refine_5
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ Eq (Finset.univ.sum fun i => HSMul.hSMul (Function.comp w Subtype.val i) ↑i) x
    -/
  · change (∑ i ∈ t'.attach, (fun e => w e • e) ↑i) = x
    /-
      case intro.intro.intro.intro.intro.intro.refine_5
      𝕜 : Type u_1
      E : Type u
      inst✝² : LinearOrderedField 𝕜
      inst✝¹ : AddCommGroup E
      inst✝ : Module 𝕜 E
      s : Set E
      x : E
      t : Finset E
      ht₁ : HasSubset.Subset (↑t) s
      ht₂ : AffineIndependent 𝕜 Subtype.val
      w : E → 𝕜
      hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
      hw₂ : Eq (t.sum fun y => w y) 1
      hw₃ : Eq (t.centerMass w id) x
      t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
      ⊢ Eq (t'.attach.sum fun i => (fun e => HSMul.hSMul (w e) e) ↑i) x
    -/
    rw [Finset.sum_attach (f := fun e => w e • e), Finset.sum_filter_of_ne]
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        𝕜 : Type u_1
        E : Type u
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set E
        x : E
        t : Finset E
        ht₁ : HasSubset.Subset (↑t) s
        ht₂ : AffineIndependent 𝕜 Subtype.val
        w : E → 𝕜
        hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
        hw₂ : Eq (t.sum fun y => w y) 1
        hw₃ : Eq (t.centerMass w id) x
        t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
        ⊢ Eq (t.sum fun x => HSMul.hSMul (w x) x) x
      -/
    · rw [t.centerMass_eq_of_sum_1 id hw₂] at hw₃
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        𝕜 : Type u_1
        E : Type u
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set E
        x : E
        t : Finset E
        ht₁ : HasSubset.Subset (↑t) s
        ht₂ : AffineIndependent 𝕜 Subtype.val
        w : E → 𝕜
        hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
        hw₂ : Eq (t.sum fun y => w y) 1
        hw₃ : Eq (t.sum fun i => HSMul.hSMul (w i) (id i)) x
        t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
        ⊢ Eq (t.sum fun x => HSMul.hSMul (w x) x) x
      -/
      exact hw₃
      /-
        🎉 no goals
      -/
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        𝕜 : Type u_1
        E : Type u
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set E
        x : E
        t : Finset E
        ht₁ : HasSubset.Subset (↑t) s
        ht₂ : AffineIndependent 𝕜 Subtype.val
        w : E → 𝕜
        hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
        hw₂ : Eq (t.sum fun y => w y) 1
        hw₃ : Eq (t.centerMass w id) x
        t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
        ⊢ ∀ (x : E), Membership.mem t x → Ne (HSMul.hSMul (w x) x) 0 → Ne (w x) 0
      -/
    · intro e _ hwe contra
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        𝕜 : Type u_1
        E : Type u
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set E
        x : E
        t : Finset E
        ht₁ : HasSubset.Subset (↑t) s
        ht₂ : AffineIndependent 𝕜 Subtype.val
        w : E → 𝕜
        hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
        hw₂ : Eq (t.sum fun y => w y) 1
        hw₃ : Eq (t.centerMass w id) x
        t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
        e : E
        a✝ : Membership.mem t e
        hwe : Ne (HSMul.hSMul (w e) e) 0
        contra : Eq (w e) 0
        ⊢ False
      -/
      apply hwe
      /-
        case intro.intro.intro.intro.intro.intro.refine_5
        𝕜 : Type u_1
        E : Type u
        inst✝² : LinearOrderedField 𝕜
        inst✝¹ : AddCommGroup E
        inst✝ : Module 𝕜 E
        s : Set E
        x : E
        t : Finset E
        ht₁ : HasSubset.Subset (↑t) s
        ht₂ : AffineIndependent 𝕜 Subtype.val
        w : E → 𝕜
        hw₁ : ∀ (y : E), Membership.mem t y → LE.le 0 (w y)
        hw₂ : Eq (t.sum fun y => w y) 1
        hw₃ : Eq (t.centerMass w id) x
        t' : Finset E := Finset.filter (fun i => Ne (w i) 0) t
        e : E
        a✝ : Membership.mem t e
        hwe : Ne (HSMul.hSMul (w e) e) 0
        contra : Eq (w e) 0
        ⊢ Eq (HSMul.hSMul (w e) e) 0
      -/
      rw [contra, zero_smul]
      /-
        🎉 no goals
      -/

