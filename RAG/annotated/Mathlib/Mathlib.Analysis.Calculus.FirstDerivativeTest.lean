/-- The First-Derivative Test from calculus, maxima version.
  Suppose `a < b < c`, `f : ℝ → ℝ` is continuous at `b`,
  the derivative `f'` is nonnegative on `(a,b)`, and
  the derivative `f'` is nonpositive on `(b,c)`. Then `f` has a local maximum at `a`. -/
lemma isLocalMax_of_deriv_Ioo {f : ℝ → ℝ} {a b c : ℝ} (g₀ : a < b) (g₁ : b < c)
    (h : ContinuousAt f b)
    (hd₀ : DifferentiableOn ℝ f (Ioo a b))
    (hd₁ : DifferentiableOn ℝ f (Ioo b c))
    (h₀ : ∀ x ∈ Ioo a b, 0 ≤ deriv f x)
    (h₁ : ∀ x ∈ Ioo b c, deriv f x ≤ 0) : IsLocalMax f b :=
  have hIoc : ContinuousOn f (Ioc a b) :=
    Ioo_union_right g₀ ▸ hd₀.continuousOn.union_continuousAt (isOpen_Ioo (a := a) (b := b))
          /-
            f : Real → Real
            a b c : Real
            g₀ : LT.lt a b
            g₁ : LT.lt b c
            h : ContinuousAt f b
            hd₀ : DifferentiableOn Real f (Set.Ioo a b)
            hd₁ : DifferentiableOn Real f (Set.Ioo b c)
            h₀ : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LE.le 0 (deriv f x)
            h₁ : ∀ (x : Real), Membership.mem (Set.Ioo b c) x → LE.le (deriv f x) 0
            ⊢ ∀ (x : Real), Membership.mem (Singleton.singleton b) x → ContinuousAt f x
          -/
      (by simp_all)
          /-
            🎉 no goals
          -/
  have hIco : ContinuousOn f (Ico b c) :=
    Ioo_union_left g₁ ▸ hd₁.continuousOn.union_continuousAt (isOpen_Ioo (a := b) (b := c))
          /-
            f : Real → Real
            a b c : Real
            g₀ : LT.lt a b
            g₁ : LT.lt b c
            h : ContinuousAt f b
            hd₀ : DifferentiableOn Real f (Set.Ioo a b)
            hd₁ : DifferentiableOn Real f (Set.Ioo b c)
            h₀ : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LE.le 0 (deriv f x)
            h₁ : ∀ (x : Real), Membership.mem (Set.Ioo b c) x → LE.le (deriv f x) 0
            hIoc : ContinuousOn f (Set.Ioc a b)
            ⊢ ∀ (x : Real), Membership.mem (Singleton.singleton b) x → ContinuousAt f x
          -/
      (by simp_all)
          /-
            🎉 no goals
          -/
  isLocalMax_of_mono_anti g₀ g₁
                                                          /-
                                                            f : Real → Real
                                                            a b c : Real
                                                            g₀ : LT.lt a b
                                                            g₁ : LT.lt b c
                                                            h : ContinuousAt f b
                                                            hd₀ : DifferentiableOn Real f (Set.Ioo a b)
                                                            hd₁ : DifferentiableOn Real f (Set.Ioo b c)
                                                            h₀ : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LE.le 0 (deriv f x)
                                                            h₁ : ∀ (x : Real), Membership.mem (Set.Ioo b c) x → LE.le (deriv f x) 0
                                                            hIoc : ContinuousOn f (Set.Ioc a b)
                                                            hIco : ContinuousOn f (Set.Ico b c)
                                                            ⊢ DifferentiableOn Real f (interior (Set.Ioc a b))
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    (monotoneOn_of_deriv_nonneg (convex_Ioc a b) hIoc (by simp_all) (by simp_all))
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
                                                          /-
                                                            f : Real → Real
                                                            a b c : Real
                                                            g₀ : LT.lt a b
                                                            g₁ : LT.lt b c
                                                            h : ContinuousAt f b
                                                            hd₀ : DifferentiableOn Real f (Set.Ioo a b)
                                                            hd₁ : DifferentiableOn Real f (Set.Ioo b c)
                                                            h₀ : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LE.le 0 (deriv f x)
                                                            h₁ : ∀ (x : Real), Membership.mem (Set.Ioo b c) x → LE.le (deriv f x) 0
                                                            hIoc : ContinuousOn f (Set.Ioc a b)
                                                            hIco : ContinuousOn f (Set.Ico b c)
                                                            ⊢ DifferentiableOn Real f (interior (Set.Ico b c))
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    (antitoneOn_of_deriv_nonpos (convex_Ico b c) hIco (by simp_all) (by simp_all))
                                                                        /-
                                                                          🎉 no goals
                                                                        -/



/-- The First-Derivative Test from calculus, minima version. -/
lemma isLocalMin_of_deriv_Ioo {f : ℝ → ℝ} {a b c : ℝ}
    (g₀ : a < b) (g₁ : b < c) (h : ContinuousAt f b)
    (hd₀ : DifferentiableOn ℝ f (Ioo a b)) (hd₁ : DifferentiableOn ℝ f (Ioo b c))
    (h₀ : ∀ x ∈ Ioo a b, deriv f x ≤ 0)
    (h₁ : ∀ x ∈ Ioo b c, 0 ≤ deriv f x) : IsLocalMin f b := by
  have := isLocalMax_of_deriv_Ioo (f := -f) g₀ g₁
    (by simp_all) hd₀.neg hd₁.neg
    (fun x hx => deriv.neg (f := f) ▸ Left.nonneg_neg_iff.mpr <|h₀ x hx)
    (fun x hx => deriv.neg (f := f) ▸ Left.neg_nonpos_iff.mpr <|h₁ x hx)
  /-
    f : Real → Real
    a b c : Real
    g₀ : LT.lt a b
    g₁ : LT.lt b c
    h : ContinuousAt f b
    hd₀ : DifferentiableOn Real f (Set.Ioo a b)
    hd₁ : DifferentiableOn Real f (Set.Ioo b c)
    h₀ : ∀ (x : Real), Membership.mem (Set.Ioo a b) x → LE.le (deriv f x) 0
    h₁ : ∀ (x : Real), Membership.mem (Set.Ioo b c) x → LE.le 0 (deriv f x)
    this : IsLocalMax (Neg.neg f) b
    ⊢ IsLocalMin f b
  -/
  exact (neg_neg f) ▸ IsLocalMax.neg this
  /-
    🎉 no goals
  -/

 
/-- The First-Derivative Test from calculus, maxima version,
 expressed in terms of left and right filters. -/
lemma isLocalMax_of_deriv' {f : ℝ → ℝ} {b : ℝ} (h : ContinuousAt f b)
    (hd₀ : ∀ᶠ x in 𝓝[<] b, DifferentiableAt ℝ f x) (hd₁ : ∀ᶠ x in 𝓝[>] b, DifferentiableAt ℝ f x)
    (h₀ : ∀ᶠ x in 𝓝[<] b, 0 ≤ deriv f x) (h₁ : ∀ᶠ x in 𝓝[>] b, deriv f x ≤ 0) :
    IsLocalMax f b := by
  /-
    f : Real → Real
    b : Real
    h : ContinuousAt f b
    hd₀ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    hd₁ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    h₀ : Filter.Eventually (fun x => LE.le 0 (deriv f x)) (nhdsWithin b (Set.Iio b))
    h₁ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Ioi b))
    ⊢ IsLocalMax f b
  -/
  obtain ⟨a, ha⟩ := (nhdsLT_basis b).eventually_iff.mp <| hd₀.and h₀
  /-
    case intro
    f : Real → Real
    b : Real
    h : ContinuousAt f b
    hd₀ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    hd₁ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    h₀ : Filter.Eventually (fun x => LE.le 0 (deriv f x)) (nhdsWithin b (Set.Iio b))
    h₁ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Ioi b))
    a : Real
    ha : And (LT.lt a b) (∀ ⦃x : Real⦄, Membership.mem (Set.Ioo a b) x → And (Diff …
    ⊢ IsLocalMax f b
  -/
  obtain ⟨c, hc⟩ := (nhdsGT_basis b).eventually_iff.mp <| hd₁.and h₁
  exact isLocalMax_of_deriv_Ioo ha.1 hc.1 h
    (fun _ hx => (ha.2 hx).1.differentiableWithinAt)
    (fun _ hx => (hc.2 hx).1.differentiableWithinAt)
    (fun _ hx => (ha.2 hx).2) (fun x hx => (hc.2 hx).2)

 
/-- The First-Derivative Test from calculus, minima version,
 expressed in terms of left and right filters. -/
lemma isLocalMin_of_deriv' {f : ℝ → ℝ} {b : ℝ} (h : ContinuousAt f b)
    (hd₀ : ∀ᶠ x in 𝓝[<] b, DifferentiableAt ℝ f x) (hd₁ : ∀ᶠ x in 𝓝[>] b, DifferentiableAt ℝ f x)
    (h₀ : ∀ᶠ x in 𝓝[<] b, deriv f x ≤ 0) (h₁ : ∀ᶠ x in 𝓝[>] b, deriv f x ≥ 0) :
    IsLocalMin f b := by
  /-
    f : Real → Real
    b : Real
    h : ContinuousAt f b
    hd₀ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    hd₁ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    h₀ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Iio b))
    h₁ : Filter.Eventually (fun x => GE.ge (deriv f x) 0) (nhdsWithin b (Set.Ioi b))
    ⊢ IsLocalMin f b
  -/
  obtain ⟨a, ha⟩ := (nhdsLT_basis b).eventually_iff.mp <| hd₀.and h₀
  /-
    case intro
    f : Real → Real
    b : Real
    h : ContinuousAt f b
    hd₀ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    hd₁ : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Se …
    h₀ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Iio b))
    h₁ : Filter.Eventually (fun x => GE.ge (deriv f x) 0) (nhdsWithin b (Set.Ioi b))
    a : Real
    ha : And (LT.lt a b) (∀ ⦃x : Real⦄, Membership.mem (Set.Ioo a b) x → And (Diff …
    ⊢ IsLocalMin f b
  -/
  obtain ⟨c, hc⟩ := (nhdsGT_basis b).eventually_iff.mp <| hd₁.and h₁
  exact isLocalMin_of_deriv_Ioo ha.1 hc.1 h
    (fun _ hx => (ha.2 hx).1.differentiableWithinAt)
    (fun _ hx => (hc.2 hx).1.differentiableWithinAt)
    (fun _ hx => (ha.2 hx).2) (fun x hx => (hc.2 hx).2)


/-- The First Derivative test, maximum version. -/
theorem isLocalMax_of_deriv {f : ℝ → ℝ} {b : ℝ} (h : ContinuousAt f b)
    (hd : ∀ᶠ x in 𝓝[≠] b, DifferentiableAt ℝ f x)
    (h₀ : ∀ᶠ x in 𝓝[<] b, 0 ≤ deriv f x) (h₁ : ∀ᶠ x in 𝓝[>] b, deriv f x ≤ 0) :
    IsLocalMax f b :=
                                                 /-
                                                   f : Real → Real
                                                   b : Real
                                                   h : ContinuousAt f b
                                                   hd : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Has …
                                                   h₀ : Filter.Eventually (fun x => LE.le 0 (deriv f x)) (nhdsWithin b (Set.Iio b))
                                                   h₁ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Ioi b))
                                                   ⊢ Membership.mem (nhdsWithin b (HasCompl.compl (Singleton.singleton b))) (setO …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  isLocalMax_of_deriv' h (nhdsLT_le_nhdsNE _ (by tauto)) (nhdsGT_le_nhdsNE _ (by tauto)) h₀ h₁
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- The First Derivative test, minimum version. -/
theorem isLocalMin_of_deriv {f : ℝ → ℝ} {b : ℝ} (h : ContinuousAt f b)
    (hd : ∀ᶠ x in 𝓝[≠] b, DifferentiableAt ℝ f x)
    (h₀ : ∀ᶠ x in 𝓝[<] b, deriv f x ≤ 0) (h₁ : ∀ᶠ x in 𝓝[>] b, 0 ≤ deriv f x) :
    IsLocalMin f b :=
                                                 /-
                                                   f : Real → Real
                                                   b : Real
                                                   h : ContinuousAt f b
                                                   hd : Filter.Eventually (fun x => DifferentiableAt Real f x) (nhdsWithin b (Has …
                                                   h₀ : Filter.Eventually (fun x => LE.le (deriv f x) 0) (nhdsWithin b (Set.Iio b))
                                                   h₁ : Filter.Eventually (fun x => LE.le 0 (deriv f x)) (nhdsWithin b (Set.Ioi b))
                                                   ⊢ Membership.mem (nhdsWithin b (HasCompl.compl (Singleton.singleton b))) (setO …
                                                 -/
                                                 /-
                                                   🎉 no goals
                                                 -/
  isLocalMin_of_deriv' h (nhdsLT_le_nhdsNE _ (by tauto)) (nhdsGT_le_nhdsNE _ (by tauto)) h₀ h₁
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/

