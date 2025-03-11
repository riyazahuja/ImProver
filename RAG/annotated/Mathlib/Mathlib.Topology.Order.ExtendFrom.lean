theorem continuousOn_Icc_extendFrom_Ioo [TopologicalSpace α] [LinearOrder α] [DenselyOrdered α]
    [OrderTopology α] [TopologicalSpace β] [RegularSpace β] {f : α → β} {a b : α} {la lb : β}
    (hab : a ≠ b) (hf : ContinuousOn f (Ioo a b)) (ha : Tendsto f (𝓝[>] a) (𝓝 la))
    (hb : Tendsto f (𝓝[<] b) (𝓝 lb)) : ContinuousOn (extendFrom (Ioo a b) f) (Icc a b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : RegularSpace β
    f : α → β
    a b : α
    la lb : β
    hab : Ne a b
    hf : ContinuousOn f (Set.Ioo a b)
    ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
    ⊢ ContinuousOn (extendFrom (Set.Ioo a b) f) (Set.Icc a b)
  -/
  apply continuousOn_extendFrom
    /-
      case hB
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la lb : β
      hab : Ne a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      ⊢ HasSubset.Subset (Set.Icc a b) (closure (Set.Ioo a b))
    -/
  · rw [closure_Ioo hab]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la lb : β
      hab : Ne a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      ⊢ ∀ (x : α), Membership.mem (Set.Icc a b) x → Exists fun y => Filter.Tendsto f …
    -/
  · intro x x_in
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la lb : β
      hab : Ne a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      x : α
      x_in : Membership.mem (Set.Icc a b) x
      ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo a b)) (nhds y)
    -/
    rcases eq_endpoints_or_mem_Ioo_of_mem_Icc x_in with (rfl | rfl | h)
      /-
        case hf.inl
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        b : α
        la lb : β
        hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
        x : α
        hab : Ne x b
        hf : ContinuousOn f (Set.Ioo x b)
        ha : Filter.Tendsto f (nhdsWithin x (Set.Ioi x)) (nhds la)
        x_in : Membership.mem (Set.Icc x b) x
        ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo x b)) (nhds y)
      -/
    · exact ⟨la, ha.mono_left <| nhdsWithin_mono _ Ioo_subset_Ioi_self⟩
      /-
        🎉 no goals
      -/
      /-
        case hf.inr.inl
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        a : α
        la lb : β
        ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
        x : α
        hab : Ne a x
        hf : ContinuousOn f (Set.Ioo a x)
        hb : Filter.Tendsto f (nhdsWithin x (Set.Iio x)) (nhds lb)
        x_in : Membership.mem (Set.Icc a x) x
        ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo a x)) (nhds y)
      -/
    · exact ⟨lb, hb.mono_left <| nhdsWithin_mono _ Ioo_subset_Iio_self⟩
      /-
        🎉 no goals
      -/
      /-
        case hf.inr.inr
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        a b : α
        la lb : β
        hab : Ne a b
        hf : ContinuousOn f (Set.Ioo a b)
        ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
        hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
        x : α
        x_in : Membership.mem (Set.Icc a b) x
        h : Membership.mem (Set.Ioo a b) x
        ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo a b)) (nhds y)
      -/
    · exact ⟨f x, hf x h⟩
      /-
        🎉 no goals
      -/


theorem eq_lim_at_left_extendFrom_Ioo [TopologicalSpace α] [LinearOrder α] [DenselyOrdered α]
    [OrderTopology α] [TopologicalSpace β] [T2Space β] {f : α → β} {a b : α} {la : β} (hab : a < b)
    (ha : Tendsto f (𝓝[>] a) (𝓝 la)) : extendFrom (Ioo a b) f a = la := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    f : α → β
    a b : α
    la : β
    hab : LT.lt a b
    ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
    ⊢ Eq (extendFrom (Set.Ioo a b) f a) la
  -/
  apply extendFrom_eq
    /-
      case hx
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ Membership.mem (closure (Set.Ioo a b)) a
    -/
  · rw [closure_Ioo hab.ne]
    /-
      case hx
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ Membership.mem (Set.Icc a b) a
    -/
    simp only [le_of_lt hab, left_mem_Icc, right_mem_Icc]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ Filter.Tendsto f (nhdsWithin a (Set.Ioo a b)) (nhds la)
    -/
  · simpa [hab]
    /-
      🎉 no goals
    -/


theorem eq_lim_at_right_extendFrom_Ioo [TopologicalSpace α] [LinearOrder α] [DenselyOrdered α]
    [OrderTopology α] [TopologicalSpace β] [T2Space β] {f : α → β} {a b : α} {lb : β} (hab : a < b)
    (hb : Tendsto f (𝓝[<] b) (𝓝 lb)) : extendFrom (Ioo a b) f b = lb := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : T2Space β
    f : α → β
    a b : α
    lb : β
    hab : LT.lt a b
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
    ⊢ Eq (extendFrom (Set.Ioo a b) f b) lb
  -/
  apply extendFrom_eq
    /-
      case hx
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      lb : β
      hab : LT.lt a b
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      ⊢ Membership.mem (closure (Set.Ioo a b)) b
    -/
  · rw [closure_Ioo hab.ne]
    /-
      case hx
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      lb : β
      hab : LT.lt a b
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      ⊢ Membership.mem (Set.Icc a b) b
    -/
    simp only [le_of_lt hab, left_mem_Icc, right_mem_Icc]
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : T2Space β
      f : α → β
      a b : α
      lb : β
      hab : LT.lt a b
      hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
      ⊢ Filter.Tendsto f (nhdsWithin b (Set.Ioo a b)) (nhds lb)
    -/
  · simpa [hab]
    /-
      🎉 no goals
    -/


theorem continuousOn_Ico_extendFrom_Ioo [TopologicalSpace α] [LinearOrder α] [DenselyOrdered α]
    [OrderTopology α] [TopologicalSpace β] [RegularSpace β] {f : α → β} {a b : α} {la : β}
    (hab : a < b) (hf : ContinuousOn f (Ioo a b)) (ha : Tendsto f (𝓝[>] a) (𝓝 la)) :
    ContinuousOn (extendFrom (Ioo a b) f) (Ico a b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : RegularSpace β
    f : α → β
    a b : α
    la : β
    hab : LT.lt a b
    hf : ContinuousOn f (Set.Ioo a b)
    ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
    ⊢ ContinuousOn (extendFrom (Set.Ioo a b) f) (Set.Ico a b)
  -/
  apply continuousOn_extendFrom
    /-
      case hB
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ HasSubset.Subset (Set.Ico a b) (closure (Set.Ioo a b))
    -/
  · rw [closure_Ioo hab.ne]
    /-
      case hB
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ HasSubset.Subset (Set.Ico a b) (Set.Icc a b)
    -/
    exact Ico_subset_Icc_self
    /-
      🎉 no goals
    -/
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      ⊢ ∀ (x : α), Membership.mem (Set.Ico a b) x → Exists fun y => Filter.Tendsto f …
    -/
  · intro x x_in
    /-
      case hf
      α : Type u_1
      β : Type u_2
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : LinearOrder α
      inst✝³ : DenselyOrdered α
      inst✝² : OrderTopology α
      inst✝¹ : TopologicalSpace β
      inst✝ : RegularSpace β
      f : α → β
      a b : α
      la : β
      hab : LT.lt a b
      hf : ContinuousOn f (Set.Ioo a b)
      ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
      x : α
      x_in : Membership.mem (Set.Ico a b) x
      ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo a b)) (nhds y)
    -/
    rcases eq_left_or_mem_Ioo_of_mem_Ico x_in with (rfl | h)
      /-
        case hf.inl
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        b : α
        la : β
        x : α
        hab : LT.lt x b
        hf : ContinuousOn f (Set.Ioo x b)
        ha : Filter.Tendsto f (nhdsWithin x (Set.Ioi x)) (nhds la)
        x_in : Membership.mem (Set.Ico x b) x
        ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo x b)) (nhds y)
      -/
    · use la
      /-
        case h
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        b : α
        la : β
        x : α
        hab : LT.lt x b
        hf : ContinuousOn f (Set.Ioo x b)
        ha : Filter.Tendsto f (nhdsWithin x (Set.Ioi x)) (nhds la)
        x_in : Membership.mem (Set.Ico x b) x
        ⊢ Filter.Tendsto f (nhdsWithin x (Set.Ioo x b)) (nhds la)
      -/
      simpa [hab]
      /-
        🎉 no goals
      -/
      /-
        case hf.inr
        α : Type u_1
        β : Type u_2
        inst✝⁵ : TopologicalSpace α
        inst✝⁴ : LinearOrder α
        inst✝³ : DenselyOrdered α
        inst✝² : OrderTopology α
        inst✝¹ : TopologicalSpace β
        inst✝ : RegularSpace β
        f : α → β
        a b : α
        la : β
        hab : LT.lt a b
        hf : ContinuousOn f (Set.Ioo a b)
        ha : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds la)
        x : α
        x_in : Membership.mem (Set.Ico a b) x
        h : Membership.mem (Set.Ioo a b) x
        ⊢ Exists fun y => Filter.Tendsto f (nhdsWithin x (Set.Ioo a b)) (nhds y)
      -/
    · exact ⟨f x, hf x h⟩
      /-
        🎉 no goals
      -/


theorem continuousOn_Ioc_extendFrom_Ioo [TopologicalSpace α] [LinearOrder α] [DenselyOrdered α]
    [OrderTopology α] [TopologicalSpace β] [RegularSpace β] {f : α → β} {a b : α} {lb : β}
    (hab : a < b) (hf : ContinuousOn f (Ioo a b)) (hb : Tendsto f (𝓝[<] b) (𝓝 lb)) :
    ContinuousOn (extendFrom (Ioo a b) f) (Ioc a b) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : RegularSpace β
    f : α → β
    a b : α
    lb : β
    hab : LT.lt a b
    hf : ContinuousOn f (Set.Ioo a b)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
    ⊢ ContinuousOn (extendFrom (Set.Ioo a b) f) (Set.Ioc a b)
  -/
  have := @continuousOn_Ico_extendFrom_Ioo αᵒᵈ _ _ _ _ _ _ _ f _ _ lb hab
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : RegularSpace β
    f : α → β
    a b : α
    lb : β
    hab : LT.lt a b
    hf : ContinuousOn f (Set.Ioo a b)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
    this : ContinuousOn f (Set.Ioo b a) → Filter.Tendsto f (nhdsWithin b (Set.Ioi  …
    ⊢ ContinuousOn (extendFrom (Set.Ioo a b) f) (Set.Ioc a b)
  -/
  erw [dual_Ico, dual_Ioi, dual_Ioo] at this
  /-
    α : Type u_1
    β : Type u_2
    inst✝⁵ : TopologicalSpace α
    inst✝⁴ : LinearOrder α
    inst✝³ : DenselyOrdered α
    inst✝² : OrderTopology α
    inst✝¹ : TopologicalSpace β
    inst✝ : RegularSpace β
    f : α → β
    a b : α
    lb : β
    hab : LT.lt a b
    hf : ContinuousOn f (Set.Ioo a b)
    hb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds lb)
    this : ContinuousOn f (Set.preimage (⇑OrderDual.ofDual) (Set.Ioo a b)) → Filte …
    ⊢ ContinuousOn (extendFrom (Set.Ioo a b) f) (Set.Ioc a b)
  -/
  exact this hf hb
  /-
    🎉 no goals
  -/

