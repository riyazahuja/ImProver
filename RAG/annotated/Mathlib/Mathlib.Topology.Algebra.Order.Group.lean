instance (priority := 100) LinearOrderedAddCommGroup.topologicalAddGroup :
    TopologicalAddGroup G where
  continuous_add := by
    /-
      α : Type u_1
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      l : Filter α
      f g : α → G
      ⊢ Continuous fun p => HAdd.hAdd p.1 p.2
    -/
    refine continuous_iff_continuousAt.2 ?_
    /-
      α : Type u_1
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      l : Filter α
      f g : α → G
      ⊢ ∀ (x : Prod G G), ContinuousAt (fun p => HAdd.hAdd p.1 p.2) x
    -/
    rintro ⟨a, b⟩
    /-
      case mk
      α : Type u_1
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      l : Filter α
      f g : α → G
      a b : G
      ⊢ ContinuousAt (fun p => HAdd.hAdd p.1 p.2) { fst := a, snd := b }
    -/
    refine LinearOrderedAddCommGroup.tendsto_nhds.2 fun ε ε0 => ?_
    /-
      case mk
      α : Type u_1
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      l : Filter α
      f g : α → G
      a b ε : G
      ε0 : GT.gt ε 0
      ⊢ Filter.Eventually (fun b_1 => LT.lt (abs (HSub.hSub (HAdd.hAdd b_1.1 b_1.2)  …
    -/
    rcases dense_or_discrete 0 ε with (⟨δ, δ0, δε⟩ | ⟨_h₁, h₂⟩)
    · -- If there exists `δ ∈ (0, ε)`, then we choose `δ`-nhd of `a` and `(ε-δ)`-nhd of `b`
      filter_upwards [(eventually_abs_sub_lt a δ0).prod_nhds
          (eventually_abs_sub_lt b (sub_pos.2 δε))]
      /-
        case h
        α : Type u_1
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        l : Filter α
        f g : α → G
        a b ε : G
        ε0 : GT.gt ε 0
        δ : G
        δ0 : LT.lt 0 δ
        δε : LT.lt δ ε
        ⊢ ∀ (a_1 : Prod G G), And (LT.lt (abs (HSub.hSub a_1.1 a)) δ) (LT.lt (abs (HSu …
      -/
      rintro ⟨x, y⟩ ⟨hx : |x - a| < δ, hy : |y - b| < ε - δ⟩
      /-
        case h.mk.intro
        α : Type u_1
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        l : Filter α
        f g : α → G
        a b ε : G
        ε0 : GT.gt ε 0
        δ : G
        δ0 : LT.lt 0 δ
        δε : LT.lt δ ε
        x y : G
        hx : LT.lt (abs (HSub.hSub x a)) δ
        hy : LT.lt (abs (HSub.hSub y b)) (HSub.hSub ε δ)
        ⊢ LT.lt (abs (HSub.hSub (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := …
      -/
      rw [add_sub_add_comm]
      calc
        |x - a + (y - b)| ≤ |x - a| + |y - b| := abs_add _ _
        _ < δ + (ε - δ) := add_lt_add hx hy
        _ = ε := add_sub_cancel _ _
    · -- Otherwise `ε`-nhd of each point `a` is `{a}`
      have hε : ∀ {x y}, |x - y| < ε → x = y := by
        intro x y h
        simpa [sub_eq_zero] using h₂ _ h
      /-
        case mk.inr.intro
        α : Type u_1
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        l : Filter α
        f g : α → G
        a b ε : G
        ε0 : GT.gt ε 0
        _h₁ : ∀ (a : G), LT.lt 0 a → LE.le ε a
        h₂ : ∀ (a : G), LT.lt a ε → LE.le a 0
        hε : ∀ {x y : G}, LT.lt (abs (HSub.hSub x y)) ε → Eq x y
        ⊢ Filter.Eventually (fun b_1 => LT.lt (abs (HSub.hSub (HAdd.hAdd b_1.1 b_1.2)  …
      -/
      filter_upwards [(eventually_abs_sub_lt a ε0).prod_nhds (eventually_abs_sub_lt b ε0)]
      /-
        case h
        α : Type u_1
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        l : Filter α
        f g : α → G
        a b ε : G
        ε0 : GT.gt ε 0
        _h₁ : ∀ (a : G), LT.lt 0 a → LE.le ε a
        h₂ : ∀ (a : G), LT.lt a ε → LE.le a 0
        hε : ∀ {x y : G}, LT.lt (abs (HSub.hSub x y)) ε → Eq x y
        ⊢ ∀ (a_1 : Prod G G), And (LT.lt (abs (HSub.hSub a_1.1 a)) ε) (LT.lt (abs (HSu …
      -/
      rintro ⟨x, y⟩ ⟨hx : |x - a| < ε, hy : |y - b| < ε⟩
      /-
        case h.mk.intro
        α : Type u_1
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        l : Filter α
        f g : α → G
        a b ε : G
        ε0 : GT.gt ε 0
        _h₁ : ∀ (a : G), LT.lt 0 a → LE.le ε a
        h₂ : ∀ (a : G), LT.lt a ε → LE.le a 0
        hε : ∀ {x y : G}, LT.lt (abs (HSub.hSub x y)) ε → Eq x y
        x y : G
        hx : LT.lt (abs (HSub.hSub x a)) ε
        hy : LT.lt (abs (HSub.hSub y b)) ε
        ⊢ LT.lt (abs (HSub.hSub (HAdd.hAdd { fst := x, snd := y }.1 { fst := x, snd := …
      -/
      simpa [hε hx, hε hy]
      /-
        🎉 no goals
      -/
  continuous_neg :=
    continuous_iff_continuousAt.2 fun a =>
      LinearOrderedAddCommGroup.tendsto_nhds.2 fun ε ε0 =>
                                                         /-
                                                           α : Type u_1
                                                           G : Type u_2
                                                           inst✝² : TopologicalSpace G
                                                           inst✝¹ : LinearOrderedAddCommGroup G
                                                           inst✝ : OrderTopology G
                                                           l : Filter α
                                                           f g : α → G
                                                           a ε : G
                                                           ε0 : GT.gt ε 0
                                                           x : G
                                                           hx : LT.lt (abs (HSub.hSub x a)) ε
                                                           ⊢ LT.lt (abs (HSub.hSub (Neg.neg x) ((fun a => Neg.neg a) a))) ε
                                                         -/
        (eventually_abs_sub_lt a ε0).mono fun x hx => by rwa [neg_sub_neg, abs_sub_comm]
                                                         /-
                                                           🎉 no goals
                                                         -/


@[continuity]
theorem continuous_abs : Continuous (abs : G → G) :=
  continuous_id.max continuous_neg


protected theorem Filter.Tendsto.abs {a : G} (h : Tendsto f l (𝓝 a)) :
    Tendsto (fun x => |f x|) l (𝓝 |a|) :=
  (continuous_abs.tendsto _).comp h


theorem tendsto_zero_iff_abs_tendsto_zero (f : α → G) :
    Tendsto f l (𝓝 0) ↔ Tendsto (abs ∘ f) l (𝓝 0) := by
  /-
    α : Type u_1
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    l : Filter α
    f : α → G
    ⊢ Iff (Filter.Tendsto f l (nhds 0)) (Filter.Tendsto (Function.comp abs f) l (n …
  -/
  refine ⟨fun h => (abs_zero : |(0 : G)| = 0) ▸ h.abs, fun h => ?_⟩
  /-
    α : Type u_1
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    l : Filter α
    f : α → G
    h : Filter.Tendsto (Function.comp abs f) l (nhds 0)
    ⊢ Filter.Tendsto f l (nhds 0)
  -/
  have : Tendsto (fun a => -|f a|) l (𝓝 0) := (neg_zero : -(0 : G) = 0) ▸ h.neg
  exact
    tendsto_of_tendsto_of_tendsto_of_le_of_le this h (fun x => neg_abs_le <| f x) fun x =>
      le_abs_self <| f x


@[fun_prop]
protected theorem Continuous.abs (h : Continuous f) : Continuous fun x => |f x| :=
  continuous_abs.comp h


@[fun_prop]
protected theorem ContinuousAt.abs (h : ContinuousAt f a) : ContinuousAt (fun x => |f x|) a :=
  Filter.Tendsto.abs h


protected theorem ContinuousWithinAt.abs (h : ContinuousWithinAt f s a) :
    ContinuousWithinAt (fun x => |f x|) s a :=
  Filter.Tendsto.abs h


@[fun_prop]
protected theorem ContinuousOn.abs (h : ContinuousOn f s) : ContinuousOn (fun x => |f x|) s :=
  fun x hx => (h x hx).abs


theorem tendsto_abs_nhdsWithin_zero : Tendsto (abs : G → G) (𝓝[≠] 0) (𝓝[>] 0) :=
  (continuous_abs.tendsto' (0 : G) 0 abs_zero).inf <|
    tendsto_principal_principal.2 fun _x => abs_pos.2


/-- In a linearly ordered additive group, the integer multiples of an element are dense
iff they are the whole group. -/
theorem denseRange_zsmul_iff_surjective {a : G} :
    DenseRange (· • a : ℤ → G) ↔ Surjective (· • a : ℤ → G) := by
  /-
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    a : G
    ⊢ Iff (DenseRange fun x => HSMul.hSMul x a) (Function.Surjective fun x => HSMu …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ h.denseRange⟩
  /-
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    a : G
    h : DenseRange fun x => HSMul.hSMul x a
    ⊢ Function.Surjective fun x => HSMul.hSMul x a
  -/
  wlog ha₀ : 0 < a generalizing a
    /-
      case inr
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      this : ∀ {a : G}, (DenseRange fun x => HSMul.hSMul x a) → LT.lt 0 a → Function …
      ha₀ : Not (LT.lt 0 a)
      ⊢ Function.Surjective fun x => HSMul.hSMul x a
    -/
  · simp only [← range_eq_univ, DenseRange] at *
    /-
      case inr
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : Dense (Set.range fun x => HSMul.hSMul x a)
      ha₀ : Not (LT.lt 0 a)
      this : ∀ {a : G}, Dense (Set.range fun x => HSMul.hSMul x a) → LT.lt 0 a → Eq  …
      ⊢ Eq (Set.range fun x => HSMul.hSMul x a) Set.univ
    -/
    rcases (not_lt.1 ha₀).eq_or_lt with rfl | hlt
      /-
        case inr.inl
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        this : ∀ {a : G}, Dense (Set.range fun x => HSMul.hSMul x a) → LT.lt 0 a → Eq  …
        h : Dense (Set.range fun x => HSMul.hSMul x 0)
        ha₀ : Not (LT.lt 0 0)
        ⊢ Eq (Set.range fun x => HSMul.hSMul x 0) Set.univ
      -/
    · simpa only [smul_zero, range_const, dense_iff_closure_eq, closure_singleton] using h
      /-
        🎉 no goals
      -/
    · have H : range (· • -a : ℤ → G) = range (· • a : ℤ → G) := by
        simpa only [smul_neg, ← neg_smul] using neg_surjective.range_comp (· • a)
      /-
        case inr.inr
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        a : G
        h : Dense (Set.range fun x => HSMul.hSMul x a)
        ha₀ : Not (LT.lt 0 a)
        this : ∀ {a : G}, Dense (Set.range fun x => HSMul.hSMul x a) → LT.lt 0 a → Eq  …
        hlt : LT.lt a 0
        H : Eq (Set.range fun x => HSMul.hSMul x (Neg.neg a)) (Set.range fun x => HSMu …
        ⊢ Eq (Set.range fun x => HSMul.hSMul x a) Set.univ
      -/
      rw [← H]
      /-
        case inr.inr
        G : Type u_2
        inst✝² : TopologicalSpace G
        inst✝¹ : LinearOrderedAddCommGroup G
        inst✝ : OrderTopology G
        a : G
        h : Dense (Set.range fun x => HSMul.hSMul x a)
        ha₀ : Not (LT.lt 0 a)
        this : ∀ {a : G}, Dense (Set.range fun x => HSMul.hSMul x a) → LT.lt 0 a → Eq  …
        hlt : LT.lt a 0
        H : Eq (Set.range fun x => HSMul.hSMul x (Neg.neg a)) (Set.range fun x => HSMu …
        ⊢ Eq (Set.range fun x => HSMul.hSMul x (Neg.neg a)) Set.univ
      -/
                     /-
                       🎉 no goals
                     -/
      apply this <;> simpa only [H, neg_pos]
                     /-
                       🎉 no goals
                     -/
  /-
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    a : G
    h : DenseRange fun x => HSMul.hSMul x a
    ha₀ : LT.lt 0 a
    ⊢ Function.Surjective fun x => HSMul.hSMul x a
  -/
  intro b
  obtain ⟨m, hm, hm'⟩ : ∃ m : ℤ, m • a ∈ Ioo b (b + a + a) := by
    have hne : (Ioo b (b + a + a)).Nonempty := ⟨b + a, by simpa⟩
    simpa using h.exists_mem_open isOpen_Ioo hne
  /-
    case intro.intro
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    a : G
    h : DenseRange fun x => HSMul.hSMul x a
    ha₀ : LT.lt 0 a
    b : G
    m : Int
    hm : LT.lt b (HSMul.hSMul m a)
    hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
    ⊢ Exists fun a_1 => Eq ((fun x => HSMul.hSMul x a) a_1) b
  -/
  rcases eq_or_ne b ((m - 1) • a) with rfl | hne; · simp
                                                    /-
                                                      🎉 no goals
                                                    -/
  suffices (Ioo (m • a) ((m + 1) • a)).Nonempty by
    rcases h.exists_mem_open isOpen_Ioo this with ⟨l, hl⟩
    have : m < l ∧ l < m + 1 := by simpa [zsmul_lt_zsmul_iff_left ha₀] using hl
    omega
  /-
    case intro.intro.inr
    G : Type u_2
    inst✝² : TopologicalSpace G
    inst✝¹ : LinearOrderedAddCommGroup G
    inst✝ : OrderTopology G
    a : G
    h : DenseRange fun x => HSMul.hSMul x a
    ha₀ : LT.lt 0 a
    b : G
    m : Int
    hm : LT.lt b (HSMul.hSMul m a)
    hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
    hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
    ⊢ (Set.Ioo (HSMul.hSMul m a) (HSMul.hSMul (HAdd.hAdd m 1) a)).Nonempty
  -/
  rcases hne.lt_or_lt with hlt | hlt
    /-
      case intro.intro.inr.inl
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      ha₀ : LT.lt 0 a
      b : G
      m : Int
      hm : LT.lt b (HSMul.hSMul m a)
      hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
      hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
      hlt : LT.lt b (HSMul.hSMul (HSub.hSub m 1) a)
      ⊢ (Set.Ioo (HSMul.hSMul m a) (HSMul.hSMul (HAdd.hAdd m 1) a)).Nonempty
    -/
  · refine ⟨b + a + a, hm', ?_⟩
    /-
      case intro.intro.inr.inl
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      ha₀ : LT.lt 0 a
      b : G
      m : Int
      hm : LT.lt b (HSMul.hSMul m a)
      hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
      hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
      hlt : LT.lt b (HSMul.hSMul (HSub.hSub m 1) a)
      ⊢ LT.lt (HAdd.hAdd (HAdd.hAdd b a) a) (HSMul.hSMul (HAdd.hAdd m 1) a)
    -/
    simpa only [add_smul, sub_smul, one_smul, lt_sub_iff_add_lt, add_lt_add_iff_right] using hlt
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr.inr
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      ha₀ : LT.lt 0 a
      b : G
      m : Int
      hm : LT.lt b (HSMul.hSMul m a)
      hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
      hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
      hlt : LT.lt (HSMul.hSMul (HSub.hSub m 1) a) b
      ⊢ (Set.Ioo (HSMul.hSMul m a) (HSMul.hSMul (HAdd.hAdd m 1) a)).Nonempty
    -/
  · use b + a
    /-
      case h
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      ha₀ : LT.lt 0 a
      b : G
      m : Int
      hm : LT.lt b (HSMul.hSMul m a)
      hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
      hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
      hlt : LT.lt (HSMul.hSMul (HSub.hSub m 1) a) b
      ⊢ Membership.mem (Set.Ioo (HSMul.hSMul m a) (HSMul.hSMul (HAdd.hAdd m 1) a)) ( …
    -/
    simp only [mem_Ioo, add_smul, sub_smul, one_smul, add_lt_add_iff_right] at hlt ⊢
    /-
      case h
      G : Type u_2
      inst✝² : TopologicalSpace G
      inst✝¹ : LinearOrderedAddCommGroup G
      inst✝ : OrderTopology G
      a : G
      h : DenseRange fun x => HSMul.hSMul x a
      ha₀ : LT.lt 0 a
      b : G
      m : Int
      hm : LT.lt b (HSMul.hSMul m a)
      hm' : LT.lt (HSMul.hSMul m a) (HAdd.hAdd (HAdd.hAdd b a) a)
      hne : Ne b (HSMul.hSMul (HSub.hSub m 1) a)
      hlt : LT.lt (HSub.hSub (HSMul.hSMul m a) a) b
      ⊢ And (LT.lt (HSMul.hSMul m a) (HAdd.hAdd b a)) (LT.lt b (HSMul.hSMul m a))
    -/
    exact ⟨sub_lt_iff_lt_add.1 hlt, hm⟩
    /-
      🎉 no goals
    -/


/-- In a nontrivial densely linearly ordered additive group,
the integer multiples of an element can't be dense. -/
theorem not_denseRange_zsmul [Nontrivial G] [DenselyOrdered G] {a : G} :
    ¬DenseRange (· • a : ℤ → G) :=
  denseRange_zsmul_iff_surjective.not.mpr fun h ↦
    not_isAddCyclic_of_denselyOrdered G ⟨⟨a, h⟩⟩

