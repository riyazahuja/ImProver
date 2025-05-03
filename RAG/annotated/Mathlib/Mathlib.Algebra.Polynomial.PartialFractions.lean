/-- Let R be an integral domain and f, g₁, g₂ ∈ R[X]. Let g₁ and g₂ be monic and coprime.
Then, ∃ q, r₁, r₂ ∈ R[X] such that f / g₁g₂ = q + r₁/g₁ + r₂/g₂ and deg(r₁) < deg(g₁) and
deg(r₂) < deg(g₂).
-/
theorem div_eq_quo_add_rem_div_add_rem_div (f : R[X]) {g₁ g₂ : R[X]} (hg₁ : g₁.Monic)
    (hg₂ : g₂.Monic) (hcoprime : IsCoprime g₁ g₂) :
    ∃ q r₁ r₂ : R[X],
      r₁.degree < g₁.degree ∧
        r₂.degree < g₂.degree ∧ (f : K) / (↑g₁ * ↑g₂) = ↑q + ↑r₁ / ↑g₁ + ↑r₂ / ↑g₂ := by
  /-
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    hcoprime : IsCoprime g₁ g₂
    ⊢ Exists fun q => Exists fun r₁ => Exists fun r₂ => And (LT.lt r₁.degree g₁.de …
  -/
  rcases hcoprime with ⟨c, d, hcd⟩
  refine
    ⟨f * d /ₘ g₁ + f * c /ₘ g₂, f * d %ₘ g₁, f * c %ₘ g₂, degree_modByMonic_lt _ hg₁,
      degree_modByMonic_lt _ hg₂, ?_⟩
  have hg₁' : (↑g₁ : K) ≠ 0 := by
    norm_cast
    exact hg₁.ne_zero
  have hg₂' : (↑g₂ : K) ≠ 0 := by
    norm_cast
    exact hg₂.ne_zero
  /-
    case intro.intro
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    c d : Polynomial R
    hcd : Eq (HAdd.hAdd (HMul.hMul c g₁) (HMul.hMul d g₂)) 1
    hg₁' : Ne (↑g₁) 0
    hg₂' : Ne (↑g₂) 0
    ⊢ Eq (HDiv.hDiv (↑f) (HMul.hMul ↑g₁ ↑g₂)) (HAdd.hAdd (HAdd.hAdd (↑(HAdd.hAdd ( …
  -/
  have hfc := modByMonic_add_div (f * c) hg₂
  /-
    case intro.intro
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    c d : Polynomial R
    hcd : Eq (HAdd.hAdd (HMul.hMul c g₁) (HMul.hMul d g₂)) 1
    hg₁' : Ne (↑g₁) 0
    hg₂' : Ne (↑g₂) 0
    hfc : Eq (HAdd.hAdd ((HMul.hMul f c).modByMonic g₂) (HMul.hMul g₂ ((HMul.hMul  …
    ⊢ Eq (HDiv.hDiv (↑f) (HMul.hMul ↑g₁ ↑g₂)) (HAdd.hAdd (HAdd.hAdd (↑(HAdd.hAdd ( …
  -/
  have hfd := modByMonic_add_div (f * d) hg₁
  /-
    case intro.intro
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    c d : Polynomial R
    hcd : Eq (HAdd.hAdd (HMul.hMul c g₁) (HMul.hMul d g₂)) 1
    hg₁' : Ne (↑g₁) 0
    hg₂' : Ne (↑g₂) 0
    hfc : Eq (HAdd.hAdd ((HMul.hMul f c).modByMonic g₂) (HMul.hMul g₂ ((HMul.hMul  …
    hfd : Eq (HAdd.hAdd ((HMul.hMul f d).modByMonic g₁) (HMul.hMul g₁ ((HMul.hMul  …
    ⊢ Eq (HDiv.hDiv (↑f) (HMul.hMul ↑g₁ ↑g₂)) (HAdd.hAdd (HAdd.hAdd (↑(HAdd.hAdd ( …
  -/
  field_simp
  /-
    case intro.intro
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    c d : Polynomial R
    hcd : Eq (HAdd.hAdd (HMul.hMul c g₁) (HMul.hMul d g₂)) 1
    hg₁' : Ne (↑g₁) 0
    hg₂' : Ne (↑g₂) 0
    hfc : Eq (HAdd.hAdd ((HMul.hMul f c).modByMonic g₂) (HMul.hMul g₂ ((HMul.hMul  …
    hfd : Eq (HAdd.hAdd ((HMul.hMul f d).modByMonic g₁) (HMul.hMul g₁ ((HMul.hMul  …
    ⊢ Eq (↑f) (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (HAdd.hAdd ((algebraMap  …
  -/
  norm_cast
  /-
    case intro.intro
    R : Type
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type
    inst✝² : Field K
    inst✝¹ : Algebra (Polynomial R) K
    inst✝ : IsFractionRing (Polynomial R) K
    f g₁ g₂ : Polynomial R
    hg₁ : g₁.Monic
    hg₂ : g₂.Monic
    c d : Polynomial R
    hcd : Eq (HAdd.hAdd (HMul.hMul c g₁) (HMul.hMul d g₂)) 1
    hg₁' : Ne (↑g₁) 0
    hg₂' : Ne (↑g₂) 0
    hfc : Eq (HAdd.hAdd ((HMul.hMul f c).modByMonic g₂) (HMul.hMul g₂ ((HMul.hMul  …
    hfd : Eq (HAdd.hAdd ((HMul.hMul f d).modByMonic g₁) (HMul.hMul g₁ ((HMul.hMul  …
    ⊢ Eq f (HAdd.hAdd (HMul.hMul (HAdd.hAdd (HMul.hMul (HAdd.hAdd ((HMul.hMul f d) …
  -/
  linear_combination -1 * f * hcd + -1 * g₁ * hfc + -1 * g₂ * hfd
  /-
    🎉 no goals
  -/


/-- Let R be an integral domain and f ∈ R[X]. Let s be a finite index set.
Then, a fraction of the form f / ∏ (g i) can be rewritten as q + ∑ (r i) / (g i), where
deg(r i) < deg(g i), provided that the g i are monic and pairwise coprime.
-/
theorem div_eq_quo_add_sum_rem_div (f : R[X]) {ι : Type*} {g : ι → R[X]} {s : Finset ι}
    (hg : ∀ i ∈ s, (g i).Monic) (hcop : Set.Pairwise ↑s fun i j => IsCoprime (g i) (g j)) :
    ∃ (q : R[X]) (r : ι → R[X]),
      (∀ i ∈ s, (r i).degree < (g i).degree) ∧
        ((↑f : K) / ∏ i ∈ s, ↑(g i)) = ↑q + ∑ i ∈ s, (r i : K) / (g i : K) := by
  classical
  induction' s using Finset.induction_on with a b hab Hind f generalizing f
  · refine ⟨f, fun _ : ι => (0 : R[X]), fun i => ?_, by simp⟩
    rintro ⟨⟩
  obtain ⟨q₀, r₁, r₂, hdeg₁, _, hf : (↑f : K) / _ = _⟩ :=
    div_eq_quo_add_rem_div_add_rem_div R K f
      (hg a (b.mem_insert_self a) : Monic (g a))
      (monic_prod_of_monic _ _ fun i hi => hg i (Finset.mem_insert_of_mem hi) :
        Monic (∏ i ∈ b, g i))
      (IsCoprime.prod_right fun i hi =>
        hcop (Finset.mem_coe.2 (b.mem_insert_self a))
          (Finset.mem_coe.2 (Finset.mem_insert_of_mem hi)) (by rintro rfl; exact hab hi))
  obtain ⟨q, r, hrdeg, IH⟩ :=
    Hind _ (fun i hi => hg i (Finset.mem_insert_of_mem hi))
      (Set.Pairwise.mono (Finset.coe_subset.2 fun i hi => Finset.mem_insert_of_mem hi) hcop)
  refine ⟨q₀ + q, fun i => if i = a then r₁ else r i, ?_, ?_⟩
  · intro i
    dsimp only
    split_ifs with h1
    · cases h1
      intro
      exact hdeg₁
    · intro hi
      exact hrdeg i (Finset.mem_of_mem_insert_of_ne hi h1)
  norm_cast at hf IH ⊢
  rw [Finset.prod_insert hab, hf, IH, Finset.sum_insert hab, if_pos rfl]
  trans (↑(q₀ + q : R[X]) : K) + (↑r₁ / ↑(g a) + ∑ i ∈ b, (r i : K) / (g i : K))
  · push_cast
    ring
  congr 2
  refine Finset.sum_congr rfl fun x hxb => ?_
  rw [if_neg]
  rintro rfl
  exact hab hxb


