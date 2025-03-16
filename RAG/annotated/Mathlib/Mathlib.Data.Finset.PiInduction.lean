/-- General theorem for `Finset.induction_on_pi`-style induction principles. -/
theorem induction_on_pi_of_choice (r : ∀ i, α i → Finset (α i) → Prop)
    (H_ex : ∀ (i) (s : Finset (α i)), s.Nonempty → ∃ x ∈ s, r i x (s.erase x))
    {p : (∀ i, Finset (α i)) → Prop} (f : ∀ i, Finset (α i)) (h0 : p fun _ ↦ ∅)
    (step :
      ∀ (g : ∀ i, Finset (α i)) (i : ι) (x : α i),
        r i x (g i) → p g → p (update g i (insert x (g i)))) :
    p f := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    r : (i : ι) → α i → Finset (α i) → Prop
    H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
    p : ((i : ι) → Finset (α i)) → Prop
    f : (i : ι) → Finset (α i)
    h0 : p fun x => EmptyCollection.emptyCollection
    step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
    ⊢ p f
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    r : (i : ι) → α i → Finset (α i) → Prop
    H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
    p : ((i : ι) → Finset (α i)) → Prop
    f : (i : ι) → Finset (α i)
    h0 : p fun x => EmptyCollection.emptyCollection
    step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
    val✝ : Fintype ι
    ⊢ p f
  -/
  induction' hs : univ.sigma f using Finset.strongInductionOn with s ihs generalizing f; subst s
  /-
    case intro.a
    ι : Type u_1
    α : ι → Type u_2
    inst✝² : Finite ι
    inst✝¹ : DecidableEq ι
    inst✝ : (i : ι) → DecidableEq (α i)
    r : (i : ι) → α i → Finset (α i) → Prop
    H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
    p : ((i : ι) → Finset (α i)) → Prop
    h0 : p fun x => EmptyCollection.emptyCollection
    step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
    val✝ : Fintype ι
    f : (i : ι) → Finset (α i)
    ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
    ⊢ p f
  -/
  rcases eq_empty_or_nonempty (univ.sigma f) with he | hne
    /-
      case intro.a.inl
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      he : Eq (Finset.univ.sigma f) EmptyCollection.emptyCollection
      ⊢ p f
    -/
  · convert h0 using 1
    /-
      case h.e'_1
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      he : Eq (Finset.univ.sigma f) EmptyCollection.emptyCollection
      ⊢ Eq f fun x => EmptyCollection.emptyCollection
    -/
    simpa [funext_iff] using he
    /-
      🎉 no goals
    -/
    /-
      case intro.a.inr
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hne : (Finset.univ.sigma f).Nonempty
      ⊢ p f
    -/
  · rcases sigma_nonempty.1 hne with ⟨i, -, hi⟩
    /-
      case intro.a.inr.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      ⊢ p f
    -/
    rcases H_ex i (f i) hi with ⟨x, x_mem, hr⟩
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      hr : r i x ((f i).erase x)
      ⊢ p f
    -/
    set g := update f i ((f i).erase x) with hg
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      hr : r i x ((f i).erase x)
      g : (a : ι) → Finset (α a) := Function.update f i ((f i).erase x)
      hg : Eq g (Function.update f i ((f i).erase x))
      ⊢ p f
    -/
    clear_value g
    have hx' : x ∉ g i := by
      rw [hg, update_self]
      apply not_mem_erase
    rw [show f = update g i (insert x (g i)) by
      rw [hg, update_idem, update_self, insert_erase x_mem, update_eq_self]] at hr ihs ⊢
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      g : (a : ι) → Finset (α a)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hr : r i x ((Function.update g i (Insert.insert x (g i)) i).erase x)
      hg : Eq g (Function.update f i ((f i).erase x))
      hx' : Not (Membership.mem (g i) x)
      ⊢ p (Function.update g i (Insert.insert x (g i)))
    -/
    clear hg
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      g : (a : ι) → Finset (α a)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hr : r i x ((Function.update g i (Insert.insert x (g i)) i).erase x)
      hx' : Not (Membership.mem (g i) x)
      ⊢ p (Function.update g i (Insert.insert x (g i)))
    -/
    rw [update_self, erase_insert hx'] at hr
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      g : (a : ι) → Finset (α a)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hr : r i x (g i)
      hx' : Not (Membership.mem (g i) x)
      ⊢ p (Function.update g i (Insert.insert x (g i)))
    -/
    refine step _ _ _ hr (ihs (univ.sigma g) ?_ _ rfl)
    /-
      case intro.a.inr.intro.intro.intro.intro
      ι : Type u_1
      α : ι → Type u_2
      inst✝² : Finite ι
      inst✝¹ : DecidableEq ι
      inst✝ : (i : ι) → DecidableEq (α i)
      r : (i : ι) → α i → Finset (α i) → Prop
      H_ex : ∀ (i : ι) (s : Finset (α i)), s.Nonempty → Exists fun x => And (Members …
      p : ((i : ι) → Finset (α i)) → Prop
      h0 : p fun x => EmptyCollection.emptyCollection
      step : ∀ (g : (i : ι) → Finset (α i)) (i : ι) (x : α i), r i x (g i) → p g → p …
      val✝ : Fintype ι
      f : (i : ι) → Finset (α i)
      hne : (Finset.univ.sigma f).Nonempty
      i : ι
      hi : (f i).Nonempty
      x : α i
      x_mem : Membership.mem (f i) x
      g : (a : ι) → Finset (α a)
      ihs : ∀ (t : Finset (Sigma fun i => α i)), HasSSubset.SSubset t (Finset.univ.s …
      hr : r i x (g i)
      hx' : Not (Membership.mem (g i) x)
      ⊢ HasSSubset.SSubset (Finset.univ.sigma g) (Finset.univ.sigma (Function.update …
    -/
    rw [ssubset_iff_of_subset (sigma_mono (Subset.refl _) _)]
    exacts [⟨⟨i, x⟩, mem_sigma.2 ⟨mem_univ _, by simp⟩, by simp [hx']⟩,
      (@le_update_iff _ _ _ _ g g i _).2 ⟨subset_insert _ _, fun _ _ ↦ le_rfl⟩]


/-- Given a predicate on functions `∀ i, Finset (α i)` defined on a finite type, it is true on all
maps provided that it is true on `fun _ ↦ ∅` and for any function `g : ∀ i, Finset (α i)`, an index
`i : ι`, and `x ∉ g i`, `p g` implies `p (update g i (insert x (g i)))`.

See also `Finset.induction_on_pi_max` and `Finset.induction_on_pi_min` for specialized versions
that require `∀ i, LinearOrder (α i)`. -/
theorem induction_on_pi {p : (∀ i, Finset (α i)) → Prop} (f : ∀ i, Finset (α i)) (h0 : p fun _ ↦ ∅)
    (step : ∀ (g : ∀ i, Finset (α i)) (i : ι), ∀ x ∉ g i, p g → p (update g i (insert x (g i)))) :
    p f :=
  induction_on_pi_of_choice (fun _ x s ↦ x ∉ s) (fun _ s ⟨x, hx⟩ ↦ ⟨x, hx, not_mem_erase x s⟩) f
    h0 step

-- Porting note: this docstring is the exact translation of the one from mathlib3 but
-- the last sentence (here and in the next lemma) does make much sense to me...

/-- Given a predicate on functions `∀ i, Finset (α i)` defined on a finite type, it is true on all
maps provided that it is true on `fun _ ↦ ∅` and for any function `g : ∀ i, Finset (α i)`, an index
`i : ι`, and an element`x : α i` that is strictly greater than all elements of `g i`, `p g` implies
`p (update g i (insert x (g i)))`.

This lemma requires `LinearOrder` instances on all `α i`. See also `Finset.induction_on_pi` for a
version that `x ∉ g i` instead of ` does not need `∀ i, LinearOrder (α i)`. -/
theorem induction_on_pi_max [∀ i, LinearOrder (α i)] {p : (∀ i, Finset (α i)) → Prop}
    (f : ∀ i, Finset (α i)) (h0 : p fun _ ↦ ∅)
    (step :
      ∀ (g : ∀ i, Finset (α i)) (i : ι) (x : α i),
        (∀ y ∈ g i, y < x) → p g → p (update g i (insert x (g i)))) :
    p f :=
  induction_on_pi_of_choice (fun _ x s ↦ ∀ y ∈ s, y < x)
    (fun _ s hs ↦ ⟨s.max' hs, s.max'_mem hs, fun _ ↦ s.lt_max'_of_mem_erase_max' _⟩) f h0 step


/-- Given a predicate on functions `∀ i, Finset (α i)` defined on a finite type, it is true on all
maps provided that it is true on `fun _ ↦ ∅` and for any function `g : ∀ i, Finset (α i)`, an index
`i : ι`, and an element`x : α i` that is strictly less than all elements of `g i`, `p g` implies
`p (update g i (insert x (g i)))`.

This lemma requires `LinearOrder` instances on all `α i`. See also `Finset.induction_on_pi` for a
version that `x ∉ g i` instead of ` does not need `∀ i, LinearOrder (α i)`. -/
theorem induction_on_pi_min [∀ i, LinearOrder (α i)] {p : (∀ i, Finset (α i)) → Prop}
    (f : ∀ i, Finset (α i)) (h0 : p fun _ ↦ ∅)
    (step :
      ∀ (g : ∀ i, Finset (α i)) (i : ι) (x : α i),
        (∀ y ∈ g i, x < y) → p g → p (update g i (insert x (g i)))) :
    p f :=
  induction_on_pi_max (α := fun i ↦ (α i)ᵒᵈ) _ h0 step


