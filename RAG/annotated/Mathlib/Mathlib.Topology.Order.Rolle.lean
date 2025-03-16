/-- A continuous function on a closed interval with `f a = f b`
takes either its maximum or its minimum value at a point in the interior of the interval. -/
theorem exists_Ioo_extr_on_Icc (hab : a < b) (hfc : ContinuousOn f (Icc a b)) (hfI : f a = f b) :
    ∃ c ∈ Ioo a b, IsExtrOn f (Icc a b) c := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : ConditionallyCompleteLinearOrder X
    inst✝⁵ : DenselyOrdered X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : OrderTopology X
    inst✝² : LinearOrder Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : OrderTopology Y
    f : X → Y
    a b : X
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hfI : Eq (f a) (f b)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
  -/
  have ne : (Icc a b).Nonempty := nonempty_Icc.2 (le_of_lt hab)
  -- Consider absolute min and max points
  obtain ⟨c, cmem, cle⟩ : ∃ c ∈ Icc a b, ∀ x ∈ Icc a b, f c ≤ f x :=
    isCompact_Icc.exists_isMinOn ne hfc
  obtain ⟨C, Cmem, Cge⟩ : ∃ C ∈ Icc a b, ∀ x ∈ Icc a b, f x ≤ f C :=
    isCompact_Icc.exists_isMaxOn ne hfc
  /-
    case intro.intro.intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : ConditionallyCompleteLinearOrder X
    inst✝⁵ : DenselyOrdered X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : OrderTopology X
    inst✝² : LinearOrder Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : OrderTopology Y
    f : X → Y
    a b : X
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Icc a b)
    hfI : Eq (f a) (f b)
    ne : (Set.Icc a b).Nonempty
    c : X
    cmem : Membership.mem (Set.Icc a b) c
    cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
    C : X
    Cmem : Membership.mem (Set.Icc a b) C
    Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
  -/
  by_cases hc : f c = f a
    /-
      case pos
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : ConditionallyCompleteLinearOrder X
      inst✝⁵ : DenselyOrdered X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : OrderTopology X
      inst✝² : LinearOrder Y
      inst✝¹ : TopologicalSpace Y
      inst✝ : OrderTopology Y
      f : X → Y
      a b : X
      hab : LT.lt a b
      hfc : ContinuousOn f (Set.Icc a b)
      hfI : Eq (f a) (f b)
      ne : (Set.Icc a b).Nonempty
      c : X
      cmem : Membership.mem (Set.Icc a b) c
      cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
      C : X
      Cmem : Membership.mem (Set.Icc a b) C
      Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
      hc : Eq (f c) (f a)
      ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
    -/
  · by_cases hC : f C = f a
      /-
        case pos
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Eq (f C) (f a)
        ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
      -/
    · have : ∀ x ∈ Icc a b, f x = f a := fun x hx => le_antisymm (hC ▸ Cge x hx) (hc ▸ cle x hx)
      -- `f` is a constant, so we can take any point in `Ioo a b`
      /-
        case pos
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Eq (f C) (f a)
        this : ∀ (x : X), Membership.mem (Set.Icc a b) x → Eq (f x) (f a)
        ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
      -/
      rcases nonempty_Ioo.2 hab with ⟨c', hc'⟩
      /-
        case pos.intro
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Eq (f C) (f a)
        this : ∀ (x : X), Membership.mem (Set.Icc a b) x → Eq (f x) (f a)
        c' : X
        hc' : Membership.mem (Set.Ioo a b) c'
        ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
      -/
      refine ⟨c', hc', Or.inl fun x hx ↦ ?_⟩
      /-
        case pos.intro
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Eq (f C) (f a)
        this : ∀ (x : X), Membership.mem (Set.Icc a b) x → Eq (f x) (f a)
        c' : X
        hc' : Membership.mem (Set.Ioo a b) c'
        x : X
        hx : Membership.mem (Set.Icc a b) x
        ⊢ Membership.mem (setOf fun x => (fun x => LE.le (f c') (f x)) x) x
      -/
      simp only [mem_setOf_eq, this x hx, this c' (Ioo_subset_Icc_self hc'), le_rfl]
      /-
        🎉 no goals
      -/
      /-
        case neg
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Not (Eq (f C) (f a))
        ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
      -/
    · refine ⟨C, ⟨lt_of_le_of_ne Cmem.1 <| mt ?_ hC, lt_of_le_of_ne Cmem.2 <| mt ?_ hC⟩, Or.inr Cge⟩
      /-
        case neg.refine_1
        X : Type u_1
        Y : Type u_2
        inst✝⁶ : ConditionallyCompleteLinearOrder X
        inst✝⁵ : DenselyOrdered X
        inst✝⁴ : TopologicalSpace X
        inst✝³ : OrderTopology X
        inst✝² : LinearOrder Y
        inst✝¹ : TopologicalSpace Y
        inst✝ : OrderTopology Y
        f : X → Y
        a b : X
        hab : LT.lt a b
        hfc : ContinuousOn f (Set.Icc a b)
        hfI : Eq (f a) (f b)
        ne : (Set.Icc a b).Nonempty
        c : X
        cmem : Membership.mem (Set.Icc a b) c
        cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
        C : X
        Cmem : Membership.mem (Set.Icc a b) C
        Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
        hc : Eq (f c) (f a)
        hC : Not (Eq (f C) (f a))
        ⊢ Eq a C → Eq (f C) (f a)
      -/
      exacts [fun h => by rw [h], fun h => by rw [h, hfI]]
      /-
        🎉 no goals
      -/
    /-
      case neg
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : ConditionallyCompleteLinearOrder X
      inst✝⁵ : DenselyOrdered X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : OrderTopology X
      inst✝² : LinearOrder Y
      inst✝¹ : TopologicalSpace Y
      inst✝ : OrderTopology Y
      f : X → Y
      a b : X
      hab : LT.lt a b
      hfc : ContinuousOn f (Set.Icc a b)
      hfI : Eq (f a) (f b)
      ne : (Set.Icc a b).Nonempty
      c : X
      cmem : Membership.mem (Set.Icc a b) c
      cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
      C : X
      Cmem : Membership.mem (Set.Icc a b) C
      Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
      hc : Not (Eq (f c) (f a))
      ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Icc a  …
    -/
  · refine ⟨c, ⟨lt_of_le_of_ne cmem.1 <| mt ?_ hc, lt_of_le_of_ne cmem.2 <| mt ?_ hc⟩, Or.inl cle⟩
    /-
      case neg.refine_1
      X : Type u_1
      Y : Type u_2
      inst✝⁶ : ConditionallyCompleteLinearOrder X
      inst✝⁵ : DenselyOrdered X
      inst✝⁴ : TopologicalSpace X
      inst✝³ : OrderTopology X
      inst✝² : LinearOrder Y
      inst✝¹ : TopologicalSpace Y
      inst✝ : OrderTopology Y
      f : X → Y
      a b : X
      hab : LT.lt a b
      hfc : ContinuousOn f (Set.Icc a b)
      hfI : Eq (f a) (f b)
      ne : (Set.Icc a b).Nonempty
      c : X
      cmem : Membership.mem (Set.Icc a b) c
      cle : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f c) (f x)
      C : X
      Cmem : Membership.mem (Set.Icc a b) C
      Cge : ∀ (x : X), Membership.mem (Set.Icc a b) x → LE.le (f x) (f C)
      hc : Not (Eq (f c) (f a))
      ⊢ Eq a c → Eq (f c) (f a)
    -/
    exacts [fun h => by rw [h], fun h => by rw [h, hfI]]
    /-
      🎉 no goals
    -/


/-- A continuous function on a closed interval with `f a = f b`
has a local extremum at some point of the corresponding open interval. -/
theorem exists_isLocalExtr_Ioo (hab : a < b) (hfc : ContinuousOn f (Icc a b)) (hfI : f a = f b) :
    ∃ c ∈ Ioo a b, IsLocalExtr f c :=
  let ⟨c, cmem, hc⟩ := exists_Ioo_extr_on_Icc hab hfc hfI
  ⟨c, cmem, hc.isLocalExtr <| Icc_mem_nhds cmem.1 cmem.2⟩


/-- If a function `f` is continuous on an open interval
and tends to the same value at its endpoints, then it has an extremum on this open interval. -/
lemma exists_isExtrOn_Ioo_of_tendsto (hab : a < b) (hfc : ContinuousOn f (Ioo a b))
    (ha : Tendsto f (𝓝[>] a) (𝓝 l)) (hb : Tendsto f (𝓝[<] b) (𝓝 l)) :
    ∃ c ∈ Ioo a b, IsExtrOn f (Ioo a b) c := by
  /-
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : ConditionallyCompleteLinearOrder X
    inst✝⁵ : DenselyOrdered X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : OrderTopology X
    inst✝² : LinearOrder Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : OrderTopology Y
    f : X → Y
    a b : X
    l : Y
    hab : LT.lt a b
    hfc : ContinuousOn f (Set.Ioo a b)
    ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds l)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds l)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Ioo a  …
  -/
  have h : EqOn (extendFrom (Ioo a b) f) f (Ioo a b) := extendFrom_extends hfc
  obtain ⟨c, hc, hfc⟩ : ∃ c ∈ Ioo a b, IsExtrOn (extendFrom (Ioo a b) f) (Icc a b) c :=
    exists_Ioo_extr_on_Icc hab (continuousOn_Icc_extendFrom_Ioo hab.ne hfc ha hb)
      ((eq_lim_at_left_extendFrom_Ioo hab ha).trans (eq_lim_at_right_extendFrom_Ioo hab hb).symm)
  /-
    case intro.intro
    X : Type u_1
    Y : Type u_2
    inst✝⁶ : ConditionallyCompleteLinearOrder X
    inst✝⁵ : DenselyOrdered X
    inst✝⁴ : TopologicalSpace X
    inst✝³ : OrderTopology X
    inst✝² : LinearOrder Y
    inst✝¹ : TopologicalSpace Y
    inst✝ : OrderTopology Y
    f : X → Y
    a b : X
    l : Y
    hab : LT.lt a b
    hfc✝ : ContinuousOn f (Set.Ioo a b)
    ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds l)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds l)
    h : Set.EqOn (extendFrom (Set.Ioo a b) f) f (Set.Ioo a b)
    c : X
    hc : Membership.mem (Set.Ioo a b) c
    hfc : IsExtrOn (extendFrom (Set.Ioo a b) f) (Set.Icc a b) c
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (IsExtrOn f (Set.Ioo a  …
  -/
  exact ⟨c, hc, (hfc.on_subset Ioo_subset_Icc_self).congr h (h hc)⟩
  /-
    🎉 no goals
  -/


/-- If a function `f` is continuous on an open interval
and tends to the same value at its endpoints,
then it has a local extremum on this open interval. -/
lemma exists_isLocalExtr_Ioo_of_tendsto (hab : a < b) (hfc : ContinuousOn f (Ioo a b))
    (ha : Tendsto f (𝓝[>] a) (𝓝 l)) (hb : Tendsto f (𝓝[<] b) (𝓝 l)) :
    ∃ c ∈ Ioo a b, IsLocalExtr f c :=
  let ⟨c, cmem, hc⟩ := exists_isExtrOn_Ioo_of_tendsto hab hfc ha hb
  ⟨c, cmem, hc.isLocalExtr <| Ioo_mem_nhds cmem.1 cmem.2⟩

