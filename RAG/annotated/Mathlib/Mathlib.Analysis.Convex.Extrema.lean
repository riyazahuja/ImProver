/-- Helper lemma for the more general case: `IsMinOn.of_isLocalMinOn_of_convexOn`.
-/
theorem IsMinOn.of_isLocalMinOn_of_convexOn_Icc {f : ℝ → β} {a b : ℝ} (a_lt_b : a < b)
    (h_local_min : IsLocalMinOn f (Icc a b) a) (h_conv : ConvexOn ℝ (Icc a b) f) :
    IsMinOn f (Icc a b) a := by
  /-
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsLocalMinOn f (Set.Icc a b) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    ⊢ IsMinOn f (Set.Icc a b) a
  -/
  rintro c hc
  /-
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsLocalMinOn f (Set.Icc a b) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) c
  -/
  dsimp only [mem_setOf_eq]
  /-
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsLocalMinOn f (Set.Icc a b) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    ⊢ LE.le (f a) (f c)
  -/
  rw [IsLocalMinOn, nhdsWithin_Icc_eq_nhdsGE a_lt_b] at h_local_min
  /-
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsMinFilter f (nhdsWithin a (Set.Ici a)) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    ⊢ LE.le (f a) (f c)
  -/
  rcases hc.1.eq_or_lt with (rfl | a_lt_c)
    /-
      case inl
      β : Type u_2
      inst✝² : OrderedAddCommGroup β
      inst✝¹ : Module Real β
      inst✝ : OrderedSMul Real β
      f : Real → β
      a b : Real
      a_lt_b : LT.lt a b
      h_local_min : IsMinFilter f (nhdsWithin a (Set.Ici a)) a
      h_conv : ConvexOn Real (Set.Icc a b) f
      hc : Membership.mem (Set.Icc a b) a
      ⊢ LE.le (f a) (f a)
    -/
  · exact le_rfl
    /-
      🎉 no goals
    -/
  have H₁ : ∀ᶠ y in 𝓝[>] a, f a ≤ f y :=
    h_local_min.filter_mono (nhdsWithin_mono _ Ioi_subset_Ici_self)
  /-
    case inr
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsMinFilter f (nhdsWithin a (Set.Ici a)) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    a_lt_c : LT.lt a c
    H₁ : Filter.Eventually (fun y => LE.le (f a) (f y)) (nhdsWithin a (Set.Ioi a))
    ⊢ LE.le (f a) (f c)
  -/
  have H₂ : ∀ᶠ y in 𝓝[>] a, y ∈ Ioc a c := Ioc_mem_nhdsGT a_lt_c
  /-
    case inr
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsMinFilter f (nhdsWithin a (Set.Ici a)) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    a_lt_c : LT.lt a c
    H₁ : Filter.Eventually (fun y => LE.le (f a) (f y)) (nhdsWithin a (Set.Ioi a))
    H₂ : Filter.Eventually (fun y => Membership.mem (Set.Ioc a c) y) (nhdsWithin a …
    ⊢ LE.le (f a) (f c)
  -/
  rcases (H₁.and H₂).exists with ⟨y, hfy, hy_ac⟩
  /-
    case inr.intro.intro
    β : Type u_2
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    f : Real → β
    a b : Real
    a_lt_b : LT.lt a b
    h_local_min : IsMinFilter f (nhdsWithin a (Set.Ici a)) a
    h_conv : ConvexOn Real (Set.Icc a b) f
    c : Real
    hc : Membership.mem (Set.Icc a b) c
    a_lt_c : LT.lt a c
    H₁ : Filter.Eventually (fun y => LE.le (f a) (f y)) (nhdsWithin a (Set.Ioi a))
    H₂ : Filter.Eventually (fun y => Membership.mem (Set.Ioc a c) y) (nhdsWithin a …
    y : Real
    hfy : LE.le (f a) (f y)
    hy_ac : Membership.mem (Set.Ioc a c) y
    ⊢ LE.le (f a) (f c)
  -/
  rcases (Convex.mem_Ioc a_lt_c).mp hy_ac with ⟨ya, yc, ya₀, yc₀, yac, rfl⟩
  suffices ya • f a + yc • f a ≤ ya • f a + yc • f c from
    (smul_le_smul_iff_of_pos_left yc₀).1 (le_of_add_le_add_left this)
  calc
    ya • f a + yc • f a = f a := by rw [← add_smul, yac, one_smul]
    _ ≤ f (ya * a + yc * c) := hfy
    _ ≤ ya • f a + yc • f c := h_conv.2 (left_mem_Icc.2 a_lt_b.le) hc ya₀ yc₀.le yac


/-- A local minimum of a convex function is a global minimum, restricted to a set `s`.
-/
theorem IsMinOn.of_isLocalMinOn_of_convexOn {f : E → β} {a : E} (a_in_s : a ∈ s)
    (h_localmin : IsLocalMinOn f s a) (h_conv : ConvexOn ℝ s f) : IsMinOn f s a := by
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    ⊢ IsMinOn f s a
  -/
  intro x x_in_s
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    x : E
    x_in_s : Membership.mem s x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) x
  -/
  let g : ℝ →ᵃ[ℝ] E := AffineMap.lineMap a x
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    x : E
    x_in_s : Membership.mem s x
    g : AffineMap Real Real E := AffineMap.lineMap a x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) x
  -/
  have hg0 : g 0 = a := AffineMap.lineMap_apply_zero a x
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    x : E
    x_in_s : Membership.mem s x
    g : AffineMap Real Real E := AffineMap.lineMap a x
    hg0 : Eq (g 0) a
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) x
  -/
  have hg1 : g 1 = x := AffineMap.lineMap_apply_one a x
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    x : E
    x_in_s : Membership.mem s x
    g : AffineMap Real Real E := AffineMap.lineMap a x
    hg0 : Eq (g 0) a
    hg1 : Eq (g 1) x
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) x
  -/
  have hgc : Continuous g := AffineMap.lineMap_continuous
  have h_maps : MapsTo g (Icc 0 1) s := by
    simpa only [g, mapsTo', ← segment_eq_image_lineMap] using h_conv.1.segment_subset a_in_s x_in_s
  have fg_local_min_on : IsLocalMinOn (f ∘ g) (Icc 0 1) 0 := by
    rw [← hg0] at h_localmin
    exact h_localmin.comp_continuousOn h_maps hgc.continuousOn (left_mem_Icc.2 zero_le_one)
  have fg_min_on : IsMinOn (f ∘ g) (Icc 0 1 : Set ℝ) 0 := by
    refine IsMinOn.of_isLocalMinOn_of_convexOn_Icc one_pos fg_local_min_on ?_
    exact (h_conv.comp_affineMap g).subset h_maps (convex_Icc 0 1)
  /-
    E : Type u_1
    β : Type u_2
    inst✝⁷ : AddCommGroup E
    inst✝⁶ : TopologicalSpace E
    inst✝⁵ : Module Real E
    inst✝⁴ : TopologicalAddGroup E
    inst✝³ : ContinuousSMul Real E
    inst✝² : OrderedAddCommGroup β
    inst✝¹ : Module Real β
    inst✝ : OrderedSMul Real β
    s : Set E
    f : E → β
    a : E
    a_in_s : Membership.mem s a
    h_localmin : IsLocalMinOn f s a
    h_conv : ConvexOn Real s f
    x : E
    x_in_s : Membership.mem s x
    g : AffineMap Real Real E := AffineMap.lineMap a x
    hg0 : Eq (g 0) a
    hg1 : Eq (g 1) x
    hgc : Continuous ⇑g
    h_maps : Set.MapsTo (⇑g) (Set.Icc 0 1) s
    fg_local_min_on : IsLocalMinOn (Function.comp f ⇑g) (Set.Icc 0 1) 0
    fg_min_on : IsMinOn (Function.comp f ⇑g) (Set.Icc 0 1) 0
    ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f a) (f x)) x) x
  -/
  simpa only [hg0, hg1, comp_apply, mem_setOf_eq] using fg_min_on (right_mem_Icc.2 zero_le_one)
  /-
    🎉 no goals
  -/


/-- A local maximum of a concave function is a global maximum, restricted to a set `s`. -/
theorem IsMaxOn.of_isLocalMaxOn_of_concaveOn {f : E → β} {a : E} (a_in_s : a ∈ s)
    (h_localmax : IsLocalMaxOn f s a) (h_conc : ConcaveOn ℝ s f) : IsMaxOn f s a :=
  IsMinOn.of_isLocalMinOn_of_convexOn (β := βᵒᵈ) a_in_s h_localmax h_conc


/-- A local minimum of a convex function is a global minimum. -/
theorem IsMinOn.of_isLocalMin_of_convex_univ {f : E → β} {a : E} (h_local_min : IsLocalMin f a)
    (h_conv : ConvexOn ℝ univ f) : ∀ x, f a ≤ f x := fun x =>
  (IsMinOn.of_isLocalMinOn_of_convexOn (mem_univ a) (h_local_min.on univ) h_conv) (mem_univ x)


/-- A local maximum of a concave function is a global maximum. -/
theorem IsMaxOn.of_isLocalMax_of_convex_univ {f : E → β} {a : E} (h_local_max : IsLocalMax f a)
    (h_conc : ConcaveOn ℝ univ f) : ∀ x, f x ≤ f a :=
  IsMinOn.of_isLocalMin_of_convex_univ (β := βᵒᵈ) h_local_max h_conc

