/-- **Rolle's Theorem** `HasDerivAt` version -/
theorem exists_hasDerivAt_eq_zero (hab : a < b) (hfc : ContinuousOn f (Icc a b)) (hfI : f a = f b)
    (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) : ∃ c ∈ Ioo a b, f' c = 0 :=
  let ⟨c, cmem, hc⟩ := exists_isLocalExtr_Ioo hab hfc hfI
  ⟨c, cmem, hc.hasDerivAt_eq_zero <| hff' c cmem⟩


/-- **Rolle's Theorem** `deriv` version -/
theorem exists_deriv_eq_zero (hab : a < b) (hfc : ContinuousOn f (Icc a b)) (hfI : f a = f b) :
    ∃ c ∈ Ioo a b, deriv f c = 0 :=
  let ⟨c, cmem, hc⟩ := exists_isLocalExtr_Ioo hab hfc hfI
  ⟨c, cmem, hc.deriv_eq_zero⟩


/-- **Rolle's Theorem**, a version for a function on an open interval: if `f` has derivative `f'`
on `(a, b)` and has the same limit `l` at `𝓝[>] a` and `𝓝[<] b`, then `f' c = 0`
for some `c ∈ (a, b)`. -/
theorem exists_hasDerivAt_eq_zero' (hab : a < b) (hfa : Tendsto f (𝓝[>] a) (𝓝 l))
    (hfb : Tendsto f (𝓝[<] b) (𝓝 l)) (hff' : ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) :
    ∃ c ∈ Ioo a b, f' c = 0 :=
  let ⟨c, cmem, hc⟩ := exists_isLocalExtr_Ioo_of_tendsto hab
    (fun x hx ↦ (hff' x hx).continuousAt.continuousWithinAt) hfa hfb
  ⟨c, cmem, hc.hasDerivAt_eq_zero <| hff' c cmem⟩


/-- **Rolle's Theorem**, a version for a function on an open interval: if `f` has the same limit
`l` at `𝓝[>] a` and `𝓝[<] b`, then `deriv f c = 0` for some `c ∈ (a, b)`. This version
does not require differentiability of `f` because we define `deriv f c = 0` whenever `f` is not
differentiable at `c`. -/
theorem exists_deriv_eq_zero' (hab : a < b) (hfa : Tendsto f (𝓝[>] a) (𝓝 l))
    (hfb : Tendsto f (𝓝[<] b) (𝓝 l)) : ∃ c ∈ Ioo a b, deriv f c = 0 := by
  /-
    f : Real → Real
    a b l : Real
    hab : LT.lt a b
    hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds l)
    hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds l)
    ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (deriv f c) 0)
  -/
  by_cases h : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x
    /-
      case pos
      f : Real → Real
      a b l : Real
      hab : LT.lt a b
      hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds l)
      hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds l)
      h : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → DifferentiableAt Real f x
      ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (deriv f c) 0)
    -/
  · exact exists_hasDerivAt_eq_zero' hab hfa hfb fun x hx => (h x hx).hasDerivAt
    /-
      🎉 no goals
    -/
  · obtain ⟨c, hc, hcdiff⟩ : ∃ x ∈ Ioo a b, ¬DifferentiableAt ℝ f x := by
      push_neg at h; exact h
    /-
      case neg.intro.intro
      f : Real → Real
      a b l : Real
      hab : LT.lt a b
      hfa : Filter.Tendsto f (nhdsWithin a (Set.Ioi a)) (nhds l)
      hfb : Filter.Tendsto f (nhdsWithin b (Set.Iio b)) (nhds l)
      h : Not (∀ (x : Real), Membership.mem (Set.Ioo a b) x → DifferentiableAt Real  …
      c : Real
      hc : Membership.mem (Set.Ioo a b) c
      hcdiff : Not (DifferentiableAt Real f c)
      ⊢ Exists fun c => And (Membership.mem (Set.Ioo a b) c) (Eq (deriv f c) 0)
    -/
    exact ⟨c, hc, deriv_zero_of_not_differentiableAt hcdiff⟩
    /-
      🎉 no goals
    -/

