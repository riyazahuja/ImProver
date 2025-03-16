theorem lhopital_zero_right_on_Ioo (hab : a < b) (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Ioo a b, HasDerivAt g (g' x) x) (hg' : ∀ x ∈ Ioo a b, g' x ≠ 0)
    (hfa : Tendsto f (𝓝[>] a) (𝓝 0)) (hga : Tendsto g (𝓝[>] a) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  have sub : ∀ x ∈ Ioo a b, Ioo a x ⊆ Ioo a b := fun x hx =>
    Ioo_subset_Ioo (le_refl a) (le_of_lt hx.2)
  have hg : ∀ x ∈ Ioo a b, g x ≠ 0 := by
    intro x hx h
    have : Tendsto g (𝓝[<] x) (𝓝 0) := by
      rw [← h, ← nhdsWithin_Ioo_eq_nhdsLT hx.1]
      exact ((hgg' x hx).continuousAt.continuousWithinAt.mono <| sub x hx).tendsto
    obtain ⟨y, hyx, hy⟩ : ∃ c ∈ Ioo a x, g' c = 0 :=
      exists_hasDerivAt_eq_zero' hx.1 hga this fun y hy => hgg' y <| sub x hx hy
    exact hg' y (sub x hx hyx) hy
  have : ∀ x ∈ Ioo a b, ∃ c ∈ Ioo a x, f x * g' c = g x * f' c := by
    intro x hx
    rw [← sub_zero (f x), ← sub_zero (g x)]
    exact exists_ratio_hasDerivAt_eq_ratio_slope' g g' hx.1 f f' (fun y hy => hgg' y <| sub x hx hy)
      (fun y hy => hff' y <| sub x hx hy) hga hfa
      (tendsto_nhdsWithin_of_tendsto_nhds (hgg' x hx).continuousAt.tendsto)
      (tendsto_nhdsWithin_of_tendsto_nhds (hff' x hx).continuousAt.tendsto)
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    sub : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasSubset.Subset (Set.Ioo …
    hg : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g x) 0
    this : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Exists fun c => And (Mem …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  choose! c hc using this
  have : ∀ x ∈ Ioo a b, ((fun x' => f' x' / g' x') ∘ c) x = f x / g x := by
    intro x hx
    rcases hc x hx with ⟨h₁, h₂⟩
    field_simp [hg x hx, hg' (c x) ((sub x hx) h₁)]
    simp only [h₂]
    rw [mul_comm]
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    sub : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasSubset.Subset (Set.Ioo …
    hg : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g x) 0
    c : Real → Real
    hc : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (Membership.mem (Set.I …
    this : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Eq (Function.comp (fun x …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  have cmp : ∀ x ∈ Ioo a b, a < c x ∧ c x < x := fun x hx => (hc x hx).1
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    sub : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasSubset.Subset (Set.Ioo …
    hg : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g x) 0
    c : Real → Real
    hc : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (Membership.mem (Set.I …
    this : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Eq (Function.comp (fun x …
    cmp : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (LT.lt a (c x)) (LT.l …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rw [← nhdsWithin_Ioo_eq_nhdsGT hab]
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    sub : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasSubset.Subset (Set.Ioo …
    hg : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g x) 0
    c : Real → Real
    hc : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (Membership.mem (Set.I …
    this : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Eq (Function.comp (fun x …
    cmp : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (LT.lt a (c x)) (LT.l …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioo a b)) l
  -/
  apply tendsto_nhdsWithin_congr this
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    sub : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasSubset.Subset (Set.Ioo …
    hg : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g x) 0
    c : Real → Real
    hc : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (Membership.mem (Set.I …
    this : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Eq (Function.comp (fun x …
    cmp : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → And (LT.lt a (c x)) (LT.l …
    ⊢ Filter.Tendsto (Function.comp (fun x' => HDiv.hDiv (f' x') (g' x')) c) (nhds …
  -/
  apply hdiv.comp
  refine tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _
    (tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds
      (tendsto_nhdsWithin_of_tendsto_nhds tendsto_id) ?_ ?_) ?_
  all_goals
    apply eventually_nhdsWithin_of_forall
    intro x hx
    have := cmp x hx
    try simp
    linarith [this]


theorem lhopital_zero_right_on_Ico (hab : a < b) (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Ioo a b, HasDerivAt g (g' x) x) (hcf : ContinuousOn f (Ico a b))
    (hcg : ContinuousOn g (Ico a b)) (hg' : ∀ x ∈ Ioo a b, g' x ≠ 0) (hfa : f a = 0) (hga : g a = 0)
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hcf : ContinuousOn f (Set.Ico a b)
    hcg : ContinuousOn g (Set.Ico a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfa : Eq (f a) 0
    hga : Eq (g a) 0
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  refine lhopital_zero_right_on_Ioo hab hff' hgg' hg' ?_ ?_ hdiv
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      ⊢ Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    -/
  · rw [← hfa, ← nhdsWithin_Ioo_eq_nhdsGT hab]
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      ⊢ Filter.Tendsto f (nhdsWithin a (Set.Ioo a b)) (nhds (f a))
    -/
    exact ((hcf a <| left_mem_Ico.mpr hab).mono Ioo_subset_Ico_self).tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      ⊢ Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    -/
  · rw [← hga, ← nhdsWithin_Ioo_eq_nhdsGT hab]
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      ⊢ Filter.Tendsto g (nhdsWithin a (Set.Ioo a b)) (nhds (g a))
    -/
    exact ((hcg a <| left_mem_Ico.mpr hab).mono Ioo_subset_Ico_self).tendsto
    /-
      🎉 no goals
    -/


theorem lhopital_zero_left_on_Ioo (hab : a < b) (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Ioo a b, HasDerivAt g (g' x) x) (hg' : ∀ x ∈ Ioo a b, g' x ≠ 0)
    (hfb : Tendsto f (𝓝[<] b) (𝓝 0)) (hgb : Tendsto g (𝓝[<] b) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[<] b) l) :
  Tendsto (fun x => f x / g x) (𝓝[<] b) l := by
  -- Here, we essentially compose by `Neg.neg`. The following is mostly technical details.
  have hdnf : ∀ x ∈ -Ioo a b, HasDerivAt (f ∘ Neg.neg) (f' (-x) * -1) x := fun x hx =>
    comp x (hff' (-x) hx) (hasDerivAt_neg x)
  have hdng : ∀ x ∈ -Ioo a b, HasDerivAt (g ∘ Neg.neg) (g' (-x) * -1) x := fun x hx =>
    comp x (hgg' (-x) hx) (hasDerivAt_neg x)
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    hdnf : ∀ (x : Real), Membership.mem (Neg.neg (Set.Ioo a b)) x → HasDerivAt (Fu …
    hdng : ∀ (x : Real), Membership.mem (Neg.neg (Set.Ioo a b)) x → HasDerivAt (Fu …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  rw [neg_Ioo] at hdnf
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    hdng : ∀ (x : Real), Membership.mem (Neg.neg (Set.Ioo a b)) x → HasDerivAt (Fu …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  rw [neg_Ioo] at hdng
  have := lhopital_zero_right_on_Ioo (neg_lt_neg hab) hdnf hdng (by
    intro x hx h
    apply hg' _ (by rw [← neg_Ioo] at hx; exact hx)
    rwa [mul_comm, ← neg_eq_neg_one_mul, neg_eq_zero] at h)
    (hfb.comp tendsto_neg_nhdsGT_neg) (hgb.comp tendsto_neg_nhdsGT_neg)
    (by
      simp only [neg_div_neg_eq, mul_one, mul_neg]
      exact (tendsto_congr fun x => rfl).mp (hdiv.comp tendsto_neg_nhdsGT_neg))
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    this : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functio …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  have := this.comp tendsto_neg_nhdsLT
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functi …
    this : Filter.Tendsto (Function.comp (fun x => HDiv.hDiv (Function.comp f Neg. …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  unfold Function.comp at this
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    hgb : Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo (Neg.neg b) (Neg.neg a)) x → HasD …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functi …
    this : Filter.Tendsto (fun x => (fun x => HDiv.hDiv (f (Neg.neg x)) (g (Neg.ne …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  simpa only [neg_neg]
  /-
    🎉 no goals
  -/


theorem lhopital_zero_left_on_Ioc (hab : a < b) (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Ioo a b, HasDerivAt g (g' x) x) (hcf : ContinuousOn f (Ioc a b))
    (hcg : ContinuousOn g (Ioc a b)) (hg' : ∀ x ∈ Ioo a b, g' x ≠ 0) (hfb : f b = 0) (hgb : g b = 0)
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[<] b) l) :
    Tendsto (fun x => f x / g x) (𝓝[<] b) l := by
  /-
    a b : Real
    l : Filter Real
    f f' g g' : Real → Real
    hab : LT.lt a b
    hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
    hcf : ContinuousOn f (Set.Ioc a b)
    hcg : ContinuousOn g (Set.Ioc a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
    hfb : Eq (f b) 0
    hgb : Eq (g b) 0
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin b (Set.Iio b)) l
  -/
  refine lhopital_zero_left_on_Ioo hab hff' hgg' hg' ?_ ?_ hdiv
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ioc a b)
      hcg : ContinuousOn g (Set.Ioc a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfb : Eq (f b) 0
      hgb : Eq (g b) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
      ⊢ Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds 0)
    -/
  · rw [← hfb, ← nhdsWithin_Ioo_eq_nhdsLT hab]
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ioc a b)
      hcg : ContinuousOn g (Set.Ioc a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfb : Eq (f b) 0
      hgb : Eq (g b) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
      ⊢ Filter.Tendsto f (nhdsWithin b (Set.Ioo a b)) (nhds (f b))
    -/
    exact ((hcf b <| right_mem_Ioc.mpr hab).mono Ioo_subset_Ioc_self).tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ioc a b)
      hcg : ContinuousOn g (Set.Ioc a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfb : Eq (f b) 0
      hgb : Eq (g b) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
      ⊢ Filter.Tendsto g (nhdsWithin b (Set.Iio b)) (nhds 0)
    -/
  · rw [← hgb, ← nhdsWithin_Ioo_eq_nhdsLT hab]
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f f' g g' : Real → Real
      hab : LT.lt a b
      hff' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt f (f' x) x
      hgg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → HasDerivAt g (g' x) x
      hcf : ContinuousOn f (Set.Ioc a b)
      hcg : ContinuousOn g (Set.Ioc a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (g' x) 0
      hfb : Eq (f b) 0
      hgb : Eq (g b) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin b (Set.Ii …
      ⊢ Filter.Tendsto g (nhdsWithin b (Set.Ioo a b)) (nhds (g b))
    -/
    exact ((hcg b <| right_mem_Ioc.mpr hab).mono Ioo_subset_Ioc_self).tendsto
    /-
      🎉 no goals
    -/


theorem lhopital_zero_atTop_on_Ioi (hff' : ∀ x ∈ Ioi a, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Ioi a, HasDerivAt g (g' x) x) (hg' : ∀ x ∈ Ioi a, g' x ≠ 0)
    (hftop : Tendsto f atTop (𝓝 0)) (hgtop : Tendsto g atTop (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) atTop l) : Tendsto (fun x => f x / g x) atTop l := by
  obtain ⟨a', haa', ha'⟩ : ∃ a', a < a' ∧ 0 < a' := ⟨1 + max a 0,
    ⟨lt_of_le_of_lt (le_max_left a 0) (lt_one_add _),
      lt_of_le_of_lt (le_max_right a 0) (lt_one_add _)⟩⟩
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → Ne (g' x) 0
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    a' : Real
    haa' : LT.lt a a'
    ha' : LT.lt 0 a'
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  have fact1 : ∀ x : ℝ, x ∈ Ioo 0 a'⁻¹ → x ≠ 0 := fun _ hx => (ne_of_lt hx.1).symm
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → Ne (g' x) 0
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    a' : Real
    haa' : LT.lt a a'
    ha' : LT.lt 0 a'
    fact1 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → Ne x 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  have fact2 (x) (hx : x ∈ Ioo 0 a'⁻¹) : a < x⁻¹ := lt_trans haa' ((lt_inv_comm₀ ha' hx.1).mpr hx.2)
  have hdnf : ∀ x ∈ Ioo 0 a'⁻¹, HasDerivAt (f ∘ Inv.inv) (f' x⁻¹ * -(x ^ 2)⁻¹) x := fun x hx =>
    comp x (hff' x⁻¹ <| fact2 x hx) (hasDerivAt_inv <| fact1 x hx)
  have hdng : ∀ x ∈ Ioo 0 a'⁻¹, HasDerivAt (g ∘ Inv.inv) (g' x⁻¹ * -(x ^ 2)⁻¹) x := fun x hx =>
    comp x (hgg' x⁻¹ <| fact2 x hx) (hasDerivAt_inv <| fact1 x hx)
  have := lhopital_zero_right_on_Ioo (inv_pos.mpr ha') hdnf hdng
    (by
      intro x hx
      refine mul_ne_zero ?_ (neg_ne_zero.mpr <| inv_ne_zero <| pow_ne_zero _ <| fact1 x hx)
      exact hg' _ (fact2 x hx))
    (hftop.comp tendsto_inv_nhdsGT_zero) (hgtop.comp tendsto_inv_nhdsGT_zero)
    (by
      refine (tendsto_congr' ?_).mp (hdiv.comp tendsto_inv_nhdsGT_zero)
      rw [eventuallyEq_iff_exists_mem]
      use Ioi 0, self_mem_nhdsWithin
      intro x hx
      unfold Function.comp
      simp only
      rw [mul_div_mul_right]
      exact neg_ne_zero.mpr (inv_ne_zero <| pow_ne_zero _ <| ne_of_gt hx))
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → Ne (g' x) 0
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    a' : Real
    haa' : LT.lt a a'
    ha' : LT.lt 0 a'
    fact1 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → Ne x 0
    fact2 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → LT.lt a (Inv …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    this : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Inv.inv x) (Functio …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  have := this.comp tendsto_inv_atTop_nhdsGT_zero
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → Ne (g' x) 0
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    a' : Real
    haa' : LT.lt a a'
    ha' : LT.lt 0 a'
    fact1 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → Ne x 0
    fact2 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → LT.lt a (Inv …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Inv.inv x) (Functi …
    this : Filter.Tendsto (Function.comp (fun x => HDiv.hDiv (Function.comp f Inv. …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  unfold Function.comp at this
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Ioi a) x → Ne (g' x) 0
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    a' : Real
    haa' : LT.lt a a'
    ha' : LT.lt 0 a'
    fact1 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → Ne x 0
    fact2 : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → LT.lt a (Inv …
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioo 0 (Inv.inv a')) x → HasDerivAt (F …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Inv.inv x) (Functi …
    this : Filter.Tendsto (fun x => (fun x => HDiv.hDiv (f (Inv.inv x)) (g (Inv.in …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  simpa only [inv_inv]
  /-
    🎉 no goals
  -/


theorem lhopital_zero_atBot_on_Iio (hff' : ∀ x ∈ Iio a, HasDerivAt f (f' x) x)
    (hgg' : ∀ x ∈ Iio a, HasDerivAt g (g' x) x) (hg' : ∀ x ∈ Iio a, g' x ≠ 0)
    (hfbot : Tendsto f atBot (𝓝 0)) (hgbot : Tendsto g atBot (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) atBot l) : Tendsto (fun x => f x / g x) atBot l := by
  -- Here, we essentially compose by `Neg.neg`. The following is mostly technical details.
  have hdnf : ∀ x ∈ -Iio a, HasDerivAt (f ∘ Neg.neg) (f' (-x) * -1) x := fun x hx =>
    comp x (hff' (-x) hx) (hasDerivAt_neg x)
  have hdng : ∀ x ∈ -Iio a, HasDerivAt (g ∘ Neg.neg) (g' (-x) * -1) x := fun x hx =>
    comp x (hgg' (-x) hx) (hasDerivAt_neg x)
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → Ne (g' x) 0
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    hdnf : ∀ (x : Real), Membership.mem (Neg.neg (Set.Iio a)) x → HasDerivAt (Func …
    hdng : ∀ (x : Real), Membership.mem (Neg.neg (Set.Iio a)) x → HasDerivAt (Func …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rw [neg_Iio] at hdnf
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → Ne (g' x) 0
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    hdng : ∀ (x : Real), Membership.mem (Neg.neg (Set.Iio a)) x → HasDerivAt (Func …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rw [neg_Iio] at hdng
  have := lhopital_zero_atTop_on_Ioi hdnf hdng
    (by
      intro x hx h
      apply hg' _ (by rw [← neg_Iio] at hx; exact hx)
      rwa [mul_comm, ← neg_eq_neg_one_mul, neg_eq_zero] at h)
    (hfbot.comp tendsto_neg_atTop_atBot) (hgbot.comp tendsto_neg_atTop_atBot)
    (by
      simp only [mul_one, mul_neg, neg_div_neg_eq]
      exact (tendsto_congr fun x => rfl).mp (hdiv.comp tendsto_neg_atTop_atBot))
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → Ne (g' x) 0
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    this : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functio …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  have := this.comp tendsto_neg_atBot_atTop
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → Ne (g' x) 0
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functi …
    this : Filter.Tendsto (Function.comp (fun x => HDiv.hDiv (Function.comp f Neg. …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  unfold Function.comp at this
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt f (f' x) x
    hgg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → HasDerivAt g (g' x) x
    hg' : ∀ (x : Real), Membership.mem (Set.Iio a) x → Ne (g' x) 0
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    hdnf : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    hdng : ∀ (x : Real), Membership.mem (Set.Ioi (Neg.neg a)) x → HasDerivAt (Func …
    this✝ : Filter.Tendsto (fun x => HDiv.hDiv (Function.comp f Neg.neg x) (Functi …
    this : Filter.Tendsto (fun x => (fun x => HDiv.hDiv (f (Neg.neg x)) (g (Neg.ne …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  simpa only [neg_neg]
  /-
    🎉 no goals
  -/


theorem lhopital_zero_right_on_Ioo (hab : a < b) (hdf : DifferentiableOn ℝ f (Ioo a b))
    (hg' : ∀ x ∈ Ioo a b, deriv g x ≠ 0) (hfa : Tendsto f (𝓝[>] a) (𝓝 0))
    (hga : Tendsto g (𝓝[>] a) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  have hdf : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x := fun x hx =>
    (hdf x hx).differentiableAt (Ioo_mem_nhds hx.1 hx.2)
  have hdg : ∀ x ∈ Ioo a b, DifferentiableAt ℝ g x := fun x hx =>
    by_contradiction fun h => hg' x hx (deriv_zero_of_not_differentiableAt h)
  exact HasDerivAt.lhopital_zero_right_on_Ioo hab (fun x hx => (hdf x hx).hasDerivAt)
    (fun x hx => (hdg x hx).hasDerivAt) hg' hfa hga hdiv


theorem lhopital_zero_right_on_Ico (hab : a < b) (hdf : DifferentiableOn ℝ f (Ioo a b))
    (hcf : ContinuousOn f (Ico a b)) (hcg : ContinuousOn g (Ico a b))
    (hg' : ∀ x ∈ Ioo a b, (deriv g) x ≠ 0) (hfa : f a = 0) (hga : g a = 0)
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  /-
    a b : Real
    l : Filter Real
    f g : Real → Real
    hab : LT.lt a b
    hdf : DifferentiableOn Real f (Set.Ioo a b)
    hcf : ContinuousOn f (Set.Ico a b)
    hcg : ContinuousOn g (Set.Ico a b)
    hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (deriv g x) 0
    hfa : Eq (f a) 0
    hga : Eq (g a) 0
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  refine lhopital_zero_right_on_Ioo hab hdf hg' ?_ ?_ hdiv
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f g : Real → Real
      hab : LT.lt a b
      hdf : DifferentiableOn Real f (Set.Ioo a b)
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (deriv g x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
      ⊢ Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    -/
  · rw [← hfa, ← nhdsWithin_Ioo_eq_nhdsGT hab]
    /-
      case refine_1
      a b : Real
      l : Filter Real
      f g : Real → Real
      hab : LT.lt a b
      hdf : DifferentiableOn Real f (Set.Ioo a b)
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (deriv g x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
      ⊢ Filter.Tendsto f (nhdsWithin a (Set.Ioo a b)) (nhds (f a))
    -/
    exact ((hcf a <| left_mem_Ico.mpr hab).mono Ioo_subset_Ico_self).tendsto
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f g : Real → Real
      hab : LT.lt a b
      hdf : DifferentiableOn Real f (Set.Ioo a b)
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (deriv g x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
      ⊢ Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    -/
  · rw [← hga, ← nhdsWithin_Ioo_eq_nhdsGT hab]
    /-
      case refine_2
      a b : Real
      l : Filter Real
      f g : Real → Real
      hab : LT.lt a b
      hdf : DifferentiableOn Real f (Set.Ioo a b)
      hcf : ContinuousOn f (Set.Ico a b)
      hcg : ContinuousOn g (Set.Ico a b)
      hg' : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → Ne (deriv g x) 0
      hfa : Eq (f a) 0
      hga : Eq (g a) 0
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
      ⊢ Filter.Tendsto g (nhdsWithin a (Set.Ioo a b)) (nhds (g a))
    -/
    exact ((hcg a <| left_mem_Ico.mpr hab).mono Ioo_subset_Ico_self).tendsto
    /-
      🎉 no goals
    -/


theorem lhopital_zero_left_on_Ioo (hab : a < b) (hdf : DifferentiableOn ℝ f (Ioo a b))
    (hg' : ∀ x ∈ Ioo a b, (deriv g) x ≠ 0) (hfb : Tendsto f (𝓝[<] b) (𝓝 0))
    (hgb : Tendsto g (𝓝[<] b) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[<] b) l) :
    Tendsto (fun x => f x / g x) (𝓝[<] b) l := by
  have hdf : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x := fun x hx =>
    (hdf x hx).differentiableAt (Ioo_mem_nhds hx.1 hx.2)
  have hdg : ∀ x ∈ Ioo a b, DifferentiableAt ℝ g x := fun x hx =>
    by_contradiction fun h => hg' x hx (deriv_zero_of_not_differentiableAt h)
  exact HasDerivAt.lhopital_zero_left_on_Ioo hab (fun x hx => (hdf x hx).hasDerivAt)
    (fun x hx => (hdg x hx).hasDerivAt) hg' hfb hgb hdiv


theorem lhopital_zero_atTop_on_Ioi (hdf : DifferentiableOn ℝ f (Ioi a))
    (hg' : ∀ x ∈ Ioi a, (deriv g) x ≠ 0) (hftop : Tendsto f atTop (𝓝 0))
    (hgtop : Tendsto g atTop (𝓝 0)) (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) atTop l) :
    Tendsto (fun x => f x / g x) atTop l := by
  have hdf : ∀ x ∈ Ioi a, DifferentiableAt ℝ f x := fun x hx =>
    (hdf x hx).differentiableAt (Ioi_mem_nhds hx)
  have hdg : ∀ x ∈ Ioi a, DifferentiableAt ℝ g x := fun x hx =>
    by_contradiction fun h => hg' x hx (deriv_zero_of_not_differentiableAt h)
  exact HasDerivAt.lhopital_zero_atTop_on_Ioi (fun x hx => (hdf x hx).hasDerivAt)
    (fun x hx => (hdg x hx).hasDerivAt) hg' hftop hgtop hdiv


theorem lhopital_zero_atBot_on_Iio (hdf : DifferentiableOn ℝ f (Iio a))
    (hg' : ∀ x ∈ Iio a, (deriv g) x ≠ 0) (hfbot : Tendsto f atBot (𝓝 0))
    (hgbot : Tendsto g atBot (𝓝 0)) (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) atBot l) :
    Tendsto (fun x => f x / g x) atBot l := by
  have hdf : ∀ x ∈ Iio a, DifferentiableAt ℝ f x := fun x hx =>
    (hdf x hx).differentiableAt (Iio_mem_nhds hx)
  have hdg : ∀ x ∈ Iio a, DifferentiableAt ℝ g x := fun x hx =>
    by_contradiction fun h => hg' x hx (deriv_zero_of_not_differentiableAt h)
  exact HasDerivAt.lhopital_zero_atBot_on_Iio (fun x hx => (hdf x hx).hasDerivAt)
    (fun x hx => (hdg x hx).hasDerivAt) hg' hfbot hgbot hdiv


/-- L'Hôpital's rule for approaching a real from the right, `HasDerivAt` version -/
theorem lhopital_zero_nhds_right (hff' : ∀ᶠ x in 𝓝[>] a, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in 𝓝[>] a, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in 𝓝[>] a, g' x ≠ 0)
    (hfa : Tendsto f (𝓝[>] a) (𝓝 0)) (hga : Tendsto g (𝓝[>] a) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhdsWithin a (Set.I …
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) (nhdsWithin a (Set.I …
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) (nhdsWithin a (Set.Ioi a))
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rw [eventually_iff_exists_mem] at *
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y …
    hgg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y …
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rcases hff' with ⟨s₁, hs₁, hff'⟩
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hgg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y …
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rcases hgg' with ⟨s₂, hs₂, hgg'⟩
  /-
    case intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Ioi a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rcases hg' with ⟨s₃, hs₃, hg'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  let s := s₁ ∩ s₂ ∩ s₃
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  have hs : s ∈ 𝓝[>] a := inter_mem (inter_mem hs₁ hs₂) hs₃
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Membership.mem (nhdsWithin a (Set.Ioi a)) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rw [mem_nhdsGT_iff_exists_Ioo_subset] at hs
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Exists fun u => And (Membership.mem (Set.Ioi a) u) (HasSubset.Subset (Set …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  rcases hs with ⟨u, hau, hu⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    u : Real
    hau : Membership.mem (Set.Ioi a) u
    hu : HasSubset.Subset (Set.Ioo a u) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  refine lhopital_zero_right_on_Ioo hau ?_ ?_ ?_ hfa hga hdiv <;>
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1
      a : Real
      l : Filter Real
      f f' g g' : Real → Real
      hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
      hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      s₁ : Set Real
      hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
      hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
      s₂ : Set Real
      hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
      hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
      s₃ : Set Real
      hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
      hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
      s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
      u : Real
      hau : Membership.mem (Set.Ioi a) u
      hu : HasSubset.Subset (Set.Ioo a u) s
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioo a u) x → HasDerivAt f (f' x) x
    -/
    intro x hx <;> apply_assumption <;>
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1.a
      a : Real
      l : Filter Real
      f f' g g' : Real → Real
      hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
      hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Io …
      s₁ : Set Real
      hs₁ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₁
      hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
      s₂ : Set Real
      hs₂ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₂
      hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
      s₃ : Set Real
      hs₃ : Membership.mem (nhdsWithin a (Set.Ioi a)) s₃
      hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
      s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
      u : Real
      hau : Membership.mem (Set.Ioi a) u
      hu : HasSubset.Subset (Set.Ioo a u) s
      x : Real
      hx : Membership.mem (Set.Ioo a u) x
      ⊢ Membership.mem s₁ x
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    first | exact (hu hx).1.1 | exact (hu hx).1.2 | exact (hu hx).2
    /-
      🎉 no goals
    -/


/-- L'Hôpital's rule for approaching a real from the left, `HasDerivAt` version -/
theorem lhopital_zero_nhds_left (hff' : ∀ᶠ x in 𝓝[<] a, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in 𝓝[<] a, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in 𝓝[<] a, g' x ≠ 0)
    (hfa : Tendsto f (𝓝[<] a) (𝓝 0)) (hga : Tendsto g (𝓝[<] a) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[<] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[<] a) l := by
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhdsWithin a (Set.I …
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) (nhdsWithin a (Set.I …
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) (nhdsWithin a (Set.Iio a))
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rw [eventually_iff_exists_mem] at *
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y …
    hgg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y …
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rcases hff' with ⟨s₁, hs₁, hff'⟩
  /-
    case intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hgg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y …
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rcases hgg' with ⟨s₂, hs₂, hgg'⟩
  /-
    case intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hg' : Exists fun v => And (Membership.mem (nhdsWithin a (Set.Iio a)) v) (∀ (y  …
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rcases hg' with ⟨s₃, hs₃, hg'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  let s := s₁ ∩ s₂ ∩ s₃
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  have hs : s ∈ 𝓝[<] a := inter_mem (inter_mem hs₁ hs₂) hs₃
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Membership.mem (nhdsWithin a (Set.Iio a)) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rw [mem_nhdsLT_iff_exists_Ioo_subset] at hs
  /-
    case intro.intro.intro.intro.intro.intro
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Exists fun l => And (Membership.mem (Set.Iio a) l) (HasSubset.Subset (Set …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  rcases hs with ⟨l, hal, hl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro
    a : Real
    l✝ : Filter Real
    f f' g g' : Real → Real
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
    s₁ : Set Real
    hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    l : Real
    hal : Membership.mem (Set.Iio a) l
    hl : HasSubset.Subset (Set.Ioo l a) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l✝
  -/
  refine lhopital_zero_left_on_Ioo hal ?_ ?_ ?_ hfa hga hdiv <;> intro x hx <;> apply_assumption <;>
    /-
      case intro.intro.intro.intro.intro.intro.intro.intro.refine_1.a
      a : Real
      l✝ : Filter Real
      f f' g g' : Real → Real
      hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
      hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (Set.Ii …
      s₁ : Set Real
      hs₁ : Membership.mem (nhdsWithin a (Set.Iio a)) s₁
      hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
      s₂ : Set Real
      hs₂ : Membership.mem (nhdsWithin a (Set.Iio a)) s₂
      hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
      s₃ : Set Real
      hs₃ : Membership.mem (nhdsWithin a (Set.Iio a)) s₃
      hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
      s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
      l : Real
      hal : Membership.mem (Set.Iio a) l
      hl : HasSubset.Subset (Set.Ioo l a) s
      x : Real
      hx : Membership.mem (Set.Ioo l a) x
      ⊢ Membership.mem s₁ x
    -/
    /-
      🎉 no goals
    -/
    /-
      🎉 no goals
    -/
    first | exact (hl hx).1.1| exact (hl hx).1.2| exact (hl hx).2
    /-
      🎉 no goals
    -/


/-- L'Hôpital's rule for approaching a real, `HasDerivAt` version. This
  does not require anything about the situation at `a` -/
theorem lhopital_zero_nhds' (hff' : ∀ᶠ x in 𝓝[≠] a, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in 𝓝[≠] a, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in 𝓝[≠] a, g' x ≠ 0)
    (hfa : Tendsto f (𝓝[≠] a) (𝓝 0)) (hga : Tendsto g (𝓝[≠] a) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝[≠] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[≠] a) l := by
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhdsWithin a (HasCo …
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) (nhdsWithin a (HasCo …
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) (nhdsWithin a (HasCompl.compl ( …
    hfa : Filter.Tendsto f (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) …
    hga : Filter.Tendsto g (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) …
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhdsWithin a (HasCom …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (HasCompl.comp …
  -/
  simp only [← Iio_union_Ioi, nhdsWithin_union, tendsto_sup, eventually_sup] at *
  exact ⟨lhopital_zero_nhds_left hff'.1 hgg'.1 hg'.1 hfa.1 hga.1 hdiv.1,
    lhopital_zero_nhds_right hff'.2 hgg'.2 hg'.2 hfa.2 hga.2 hdiv.2⟩


/-- **L'Hôpital's rule** for approaching a real, `HasDerivAt` version -/
theorem lhopital_zero_nhds (hff' : ∀ᶠ x in 𝓝 a, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in 𝓝 a, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in 𝓝 a, g' x ≠ 0)
    (hfa : Tendsto f (𝓝 a) (𝓝 0)) (hga : Tendsto g (𝓝 a) (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) (𝓝 a) l) : Tendsto (fun x => f x / g x) (𝓝[≠] a) l := by
  /-
    a : Real
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhds a)
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) (nhds a)
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) (nhds a)
    hfa : Filter.Tendsto f (nhds a) (nhds 0)
    hga : Filter.Tendsto g (nhds a) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhds a) l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (HasCompl.comp …
  -/
  apply @lhopital_zero_nhds' _ _ _ f' _ g' <;>
    (first | apply eventually_nhdsWithin_of_eventually_nhds |
                                                    /-
                                                      case hff'.h
                                                      a : Real
                                                      l : Filter Real
                                                      f f' g g' : Real → Real
                                                      hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhds a)
                                                      hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) (nhds a)
                                                      hg' : Filter.Eventually (fun x => Ne (g' x) 0) (nhds a)
                                                      hfa : Filter.Tendsto f (nhds a) (nhds 0)
                                                      hga : Filter.Tendsto g (nhds a) (nhds 0)
                                                      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) (nhds a) l
                                                      ⊢ Filter.Eventually (fun x => HasDerivAt f (f' x) x) (nhds a)
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      apply tendsto_nhdsWithin_of_tendsto_nhds) <;> assumption
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- L'Hôpital's rule for approaching +∞, `HasDerivAt` version -/
theorem lhopital_zero_atTop (hff' : ∀ᶠ x in atTop, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in atTop, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in atTop, g' x ≠ 0)
    (hftop : Tendsto f atTop (𝓝 0)) (hgtop : Tendsto g atTop (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) atTop l) : Tendsto (fun x => f x / g x) atTop l := by
  /-
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) Filter.atTop
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) Filter.atTop
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) Filter.atTop
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rw [eventually_iff_exists_mem] at *
  /-
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Memb …
    hgg' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Memb …
    hg' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Membe …
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rcases hff' with ⟨s₁, hs₁, hff'⟩
  /-
    case intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hgg' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Memb …
    hg' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Membe …
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rcases hgg' with ⟨s₂, hs₂, hgg'⟩
  /-
    case intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hg' : Exists fun v => And (Membership.mem Filter.atTop v) (∀ (y : Real), Membe …
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rcases hg' with ⟨s₃, hs₃, hg'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  let s := s₁ ∩ s₂ ∩ s₃
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  have hs : s ∈ atTop := inter_mem (inter_mem hs₁ hs₂) hs₃
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Membership.mem Filter.atTop s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rw [mem_atTop_sets] at hs
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Exists fun a => ∀ (b : Real), GE.ge b a → Membership.mem s b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  rcases hs with ⟨l, hl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    l✝ : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l✝
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    l : Real
    hl : ∀ (b : Real), GE.ge b l → Membership.mem s b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l✝
  -/
  have hl' : Ioi l ⊆ s := fun x hx => hl x (le_of_lt hx)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    l✝ : Filter Real
    f f' g g' : Real → Real
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l✝
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atTop s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atTop s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atTop s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    l : Real
    hl : ∀ (b : Real), GE.ge b l → Membership.mem s b
    hl' : HasSubset.Subset (Set.Ioi l) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l✝
  -/
  refine lhopital_zero_atTop_on_Ioi ?_ ?_ (fun x hx => hg' x <| (hl' hx).2) hftop hgtop hdiv <;>
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      l✝ : Filter Real
      f f' g g' : Real → Real
      hftop : Filter.Tendsto f Filter.atTop (nhds 0)
      hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atTop l✝
      s₁ : Set Real
      hs₁ : Membership.mem Filter.atTop s₁
      hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
      s₂ : Set Real
      hs₂ : Membership.mem Filter.atTop s₂
      hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
      s₃ : Set Real
      hs₃ : Membership.mem Filter.atTop s₃
      hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
      s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
      l : Real
      hl : ∀ (b : Real), GE.ge b l → Membership.mem s b
      hl' : HasSubset.Subset (Set.Ioi l) s
      ⊢ ∀ (x : Real), Membership.mem (Set.Ioi l) x → HasDerivAt f (f' x) x
    -/
                                        /-
                                          🎉 no goals
                                        -/
    intro x hx <;> apply_assumption <;> first | exact (hl' hx).1.1| exact (hl' hx).1.2
                                        /-
                                          🎉 no goals
                                        -/


/-- L'Hôpital's rule for approaching -∞, `HasDerivAt` version -/
theorem lhopital_zero_atBot (hff' : ∀ᶠ x in atBot, HasDerivAt f (f' x) x)
    (hgg' : ∀ᶠ x in atBot, HasDerivAt g (g' x) x) (hg' : ∀ᶠ x in atBot, g' x ≠ 0)
    (hfbot : Tendsto f atBot (𝓝 0)) (hgbot : Tendsto g atBot (𝓝 0))
    (hdiv : Tendsto (fun x => f' x / g' x) atBot l) : Tendsto (fun x => f x / g x) atBot l := by
  /-
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Filter.Eventually (fun x => HasDerivAt f (f' x) x) Filter.atBot
    hgg' : Filter.Eventually (fun x => HasDerivAt g (g' x) x) Filter.atBot
    hg' : Filter.Eventually (fun x => Ne (g' x) 0) Filter.atBot
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rw [eventually_iff_exists_mem] at *
  /-
    l : Filter Real
    f f' g g' : Real → Real
    hff' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Memb …
    hgg' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Memb …
    hg' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Membe …
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rcases hff' with ⟨s₁, hs₁, hff'⟩
  /-
    case intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hgg' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Memb …
    hg' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Membe …
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rcases hgg' with ⟨s₂, hs₂, hgg'⟩
  /-
    case intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hg' : Exists fun v => And (Membership.mem Filter.atBot v) (∀ (y : Real), Membe …
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rcases hg' with ⟨s₃, hs₃, hg'⟩
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  let s := s₁ ∩ s₂ ∩ s₃
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  have hs : s ∈ atBot := inter_mem (inter_mem hs₁ hs₂) hs₃
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Membership.mem Filter.atBot s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rw [mem_atBot_sets] at hs
  /-
    case intro.intro.intro.intro.intro.intro
    l : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    hs : Exists fun a => ∀ (b : Real), LE.le b a → Membership.mem s b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  rcases hs with ⟨l, hl⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    l✝ : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l✝
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    l : Real
    hl : ∀ (b : Real), LE.le b l → Membership.mem s b
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l✝
  -/
  have hl' : Iio l ⊆ s := fun x hx => hl x (le_of_lt hx)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    l✝ : Filter Real
    f f' g g' : Real → Real
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l✝
    s₁ : Set Real
    hs₁ : Membership.mem Filter.atBot s₁
    hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
    s₂ : Set Real
    hs₂ : Membership.mem Filter.atBot s₂
    hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
    s₃ : Set Real
    hs₃ : Membership.mem Filter.atBot s₃
    hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
    s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
    l : Real
    hl : ∀ (b : Real), LE.le b l → Membership.mem s b
    hl' : HasSubset.Subset (Set.Iio l) s
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l✝
  -/
  refine lhopital_zero_atBot_on_Iio ?_ ?_ (fun x hx => hg' x <| (hl' hx).2) hfbot hgbot hdiv <;>
    /-
      case intro.intro.intro.intro.intro.intro.intro.refine_1
      l✝ : Filter Real
      f f' g g' : Real → Real
      hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
      hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (f' x) (g' x)) Filter.atBot l✝
      s₁ : Set Real
      hs₁ : Membership.mem Filter.atBot s₁
      hff' : ∀ (y : Real), Membership.mem s₁ y → HasDerivAt f (f' y) y
      s₂ : Set Real
      hs₂ : Membership.mem Filter.atBot s₂
      hgg' : ∀ (y : Real), Membership.mem s₂ y → HasDerivAt g (g' y) y
      s₃ : Set Real
      hs₃ : Membership.mem Filter.atBot s₃
      hg' : ∀ (y : Real), Membership.mem s₃ y → Ne (g' y) 0
      s : Set Real := Inter.inter (Inter.inter s₁ s₂) s₃
      l : Real
      hl : ∀ (b : Real), LE.le b l → Membership.mem s b
      hl' : HasSubset.Subset (Set.Iio l) s
      ⊢ ∀ (x : Real), Membership.mem (Set.Iio l) x → HasDerivAt f (f' x) x
    -/
                                        /-
                                          🎉 no goals
                                        -/
    intro x hx <;> apply_assumption <;> first | exact (hl' hx).1.1| exact (hl' hx).1.2
                                        /-
                                          🎉 no goals
                                        -/


/-- **L'Hôpital's rule** for approaching a real from the right, `deriv` version -/
theorem lhopital_zero_nhds_right (hdf : ∀ᶠ x in 𝓝[>] a, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x in 𝓝[>] a, deriv g x ≠ 0) (hfa : Tendsto f (𝓝[>] a) (𝓝 0))
    (hga : Tendsto g (𝓝[>] a) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[>] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[>] a) l := by
  have hdg : ∀ᶠ x in 𝓝[>] a, DifferentiableAt ℝ g x :=
    hg'.mp (Eventually.of_forall fun _ hg' =>
      by_contradiction fun h => hg' (deriv_zero_of_not_differentiableAt h))
  have hdf' : ∀ᶠ x in 𝓝[>] a, HasDerivAt f (deriv f x) x :=
    hdf.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  have hdg' : ∀ᶠ x in 𝓝[>] a, HasDerivAt g (deriv g x) x :=
    hdg.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  /-
    a : Real
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin a (Se …
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) (nhdsWithin a (Set.Ioi a))
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Ioi a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
    hdg : Filter.Eventually (fun x => DifferentiableAt Real g x) (nhdsWithin a (Se …
    hdf' : Filter.Eventually (fun x => HasDerivAt f (deriv f x) x) (nhdsWithin a ( …
    hdg' : Filter.Eventually (fun x => HasDerivAt g (deriv g x) x) (nhdsWithin a ( …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Ioi a)) l
  -/
  exact HasDerivAt.lhopital_zero_nhds_right hdf' hdg' hg' hfa hga hdiv
  /-
    🎉 no goals
  -/


/-- **L'Hôpital's rule** for approaching a real from the left, `deriv` version -/
theorem lhopital_zero_nhds_left (hdf : ∀ᶠ x in 𝓝[<] a, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x in 𝓝[<] a, deriv g x ≠ 0) (hfa : Tendsto f (𝓝[<] a) (𝓝 0))
    (hga : Tendsto g (𝓝[<] a) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[<] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[<] a) l := by
  have hdg : ∀ᶠ x in 𝓝[<] a, DifferentiableAt ℝ g x :=
    hg'.mp (Eventually.of_forall fun _ hg' =>
      by_contradiction fun h => hg' (deriv_zero_of_not_differentiableAt h))
  have hdf' : ∀ᶠ x in 𝓝[<] a, HasDerivAt f (deriv f x) x :=
    hdf.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  have hdg' : ∀ᶠ x in 𝓝[<] a, HasDerivAt g (deriv g x) x :=
    hdg.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  /-
    a : Real
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin a (Se …
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) (nhdsWithin a (Set.Iio a))
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Iio a)) (nhds 0)
    hga : Filter.Tendsto g (nhdsWithin a (Set.Iio a)) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
    hdg : Filter.Eventually (fun x => DifferentiableAt Real g x) (nhdsWithin a (Se …
    hdf' : Filter.Eventually (fun x => HasDerivAt f (deriv f x) x) (nhdsWithin a ( …
    hdg' : Filter.Eventually (fun x => HasDerivAt g (deriv g x) x) (nhdsWithin a ( …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (Set.Iio a)) l
  -/
  exact HasDerivAt.lhopital_zero_nhds_left hdf' hdg' hg' hfa hga hdiv
  /-
    🎉 no goals
  -/


/-- **L'Hôpital's rule** for approaching a real, `deriv` version. This
  does not require anything about the situation at `a` -/
theorem lhopital_zero_nhds' (hdf : ∀ᶠ x in 𝓝[≠] a, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x in 𝓝[≠] a, deriv g x ≠ 0) (hfa : Tendsto f (𝓝[≠] a) (𝓝 0))
    (hga : Tendsto g (𝓝[≠] a) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝[≠] a) l) :
    Tendsto (fun x => f x / g x) (𝓝[≠] a) l := by
  /-
    a : Real
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin a (Ha …
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) (nhdsWithin a (HasCompl.co …
    hfa : Filter.Tendsto f (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) …
    hga : Filter.Tendsto g (nhdsWithin a (HasCompl.compl (Singleton.singleton a))) …
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhdsWithin …
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (HasCompl.comp …
  -/
  simp only [← Iio_union_Ioi, nhdsWithin_union, tendsto_sup, eventually_sup] at *
  exact ⟨lhopital_zero_nhds_left hdf.1 hg'.1 hfa.1 hga.1 hdiv.1,
    lhopital_zero_nhds_right hdf.2 hg'.2 hfa.2 hga.2 hdiv.2⟩


/-- **L'Hôpital's rule** for approaching a real, `deriv` version -/
theorem lhopital_zero_nhds (hdf : ∀ᶠ x in 𝓝 a, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x in 𝓝 a, deriv g x ≠ 0) (hfa : Tendsto f (𝓝 a) (𝓝 0)) (hga : Tendsto g (𝓝 a) (𝓝 0))
    (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) (𝓝 a) l) :
    Tendsto (fun x => f x / g x) (𝓝[≠] a) l := by
  /-
    a : Real
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhds a)
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) (nhds a)
    hfa : Filter.Tendsto f (nhds a) (nhds 0)
    hga : Filter.Tendsto g (nhds a) (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhds a) l
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) (nhdsWithin a (HasCompl.comp …
  -/
  apply lhopital_zero_nhds' <;>
    (first | apply eventually_nhdsWithin_of_eventually_nhds |
                                                    /-
                                                      case hdf.h
                                                      a : Real
                                                      l : Filter Real
                                                      f g : Real → Real
                                                      hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhds a)
                                                      hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) (nhds a)
                                                      hfa : Filter.Tendsto f (nhds a) (nhds 0)
                                                      hga : Filter.Tendsto g (nhds a) (nhds 0)
                                                      hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) (nhds a) l
                                                      ⊢ Filter.Eventually (fun x => DifferentiableAt Real f x) (nhds a)
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      apply tendsto_nhdsWithin_of_tendsto_nhds) <;> assumption
                                                    /-
                                                      🎉 no goals
                                                    -/


/-- **L'Hôpital's rule** for approaching +∞, `deriv` version -/
theorem lhopital_zero_atTop (hdf : ∀ᶠ x : ℝ in atTop, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x : ℝ in atTop, deriv g x ≠ 0) (hftop : Tendsto f atTop (𝓝 0))
    (hgtop : Tendsto g atTop (𝓝 0)) (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) atTop l) :
    Tendsto (fun x => f x / g x) atTop l := by
  have hdg : ∀ᶠ x in atTop, DifferentiableAt ℝ g x := hg'.mp
    (Eventually.of_forall fun _ hg' =>
      by_contradiction fun h => hg' (deriv_zero_of_not_differentiableAt h))
  have hdf' : ∀ᶠ x in atTop, HasDerivAt f (deriv f x) x :=
    hdf.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  have hdg' : ∀ᶠ x in atTop, HasDerivAt g (deriv g x) x :=
    hdg.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  /-
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) Filter.atTop
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) Filter.atTop
    hftop : Filter.Tendsto f Filter.atTop (nhds 0)
    hgtop : Filter.Tendsto g Filter.atTop (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) Filter.atTo …
    hdg : Filter.Eventually (fun x => DifferentiableAt Real g x) Filter.atTop
    hdf' : Filter.Eventually (fun x => HasDerivAt f (deriv f x) x) Filter.atTop
    hdg' : Filter.Eventually (fun x => HasDerivAt g (deriv g x) x) Filter.atTop
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atTop l
  -/
  exact HasDerivAt.lhopital_zero_atTop hdf' hdg' hg' hftop hgtop hdiv
  /-
    🎉 no goals
  -/


/-- **L'Hôpital's rule** for approaching -∞, `deriv` version -/
theorem lhopital_zero_atBot (hdf : ∀ᶠ x : ℝ in atBot, DifferentiableAt ℝ f x)
    (hg' : ∀ᶠ x : ℝ in atBot, deriv g x ≠ 0) (hfbot : Tendsto f atBot (𝓝 0))
    (hgbot : Tendsto g atBot (𝓝 0)) (hdiv : Tendsto (fun x => (deriv f) x / (deriv g) x) atBot l) :
    Tendsto (fun x => f x / g x) atBot l := by
  have hdg : ∀ᶠ x in atBot, DifferentiableAt ℝ g x :=
    hg'.mp (Eventually.of_forall fun _ hg' =>
      by_contradiction fun h => hg' (deriv_zero_of_not_differentiableAt h))
  have hdf' : ∀ᶠ x in atBot, HasDerivAt f (deriv f x) x :=
    hdf.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  have hdg' : ∀ᶠ x in atBot, HasDerivAt g (deriv g x) x :=
    hdg.mp (Eventually.of_forall fun _ => DifferentiableAt.hasDerivAt)
  /-
    l : Filter Real
    f g : Real → Real
    hdf : Filter.Eventually (fun x => DifferentiableAt Real f x) Filter.atBot
    hg' : Filter.Eventually (fun x => Ne (deriv g x) 0) Filter.atBot
    hfbot : Filter.Tendsto f Filter.atBot (nhds 0)
    hgbot : Filter.Tendsto g Filter.atBot (nhds 0)
    hdiv : Filter.Tendsto (fun x => HDiv.hDiv (deriv f x) (deriv g x)) Filter.atBo …
    hdg : Filter.Eventually (fun x => DifferentiableAt Real g x) Filter.atBot
    hdf' : Filter.Eventually (fun x => HasDerivAt f (deriv f x) x) Filter.atBot
    hdg' : Filter.Eventually (fun x => HasDerivAt g (deriv g x) x) Filter.atBot
    ⊢ Filter.Tendsto (fun x => HDiv.hDiv (f x) (g x)) Filter.atBot l
  -/
  exact HasDerivAt.lhopital_zero_atBot hdf' hdg' hg' hfbot hgbot hdiv
  /-
    🎉 no goals
  -/


